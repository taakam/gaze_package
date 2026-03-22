"""
brisk_pose.py  (UPDATED)

Key upgrades vs previous version:
1) **Scale matching**: we resize RealSense color to ETG resolution for feature detection/matching,
   then map RS keypoints back to full-res coordinates for depth lookup + 3D backprojection.
2) **CLAHE optional** (enabled by default) to reduce exposure mismatch.
3) More robust correspondence filtering + clear debug stats.

Pose convention:
- 3D object points are in RealSense COLOR camera coordinates.
- We solve for (R,t) such that:
      x_etg ~ K_etg * (R * X_rs + t)
  i.e. transform RS_color_cam -> ETG_cam.

Usage:
    from brisk_pose import estimate_pose_brisk, BriskPnPConfig

    cfg = BriskPnPConfig()
    cfg.K_rs = rs_meta["K_color"]

    res = estimate_pose_brisk(etg_bgr, rs_color_bgr, rs_depth_m, K_etg, dist_etg, cfg=cfg)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import cv2


@dataclass
class BriskPnPConfig:
    # --- BRISK ---
    brisk_thresh: int = 20
    brisk_octaves: int = 3
    brisk_pattern_scale: float = 1.0

    # --- Preprocess ---
    use_clahe: bool = True
    clahe_clip: float = 2.0
    clahe_grid: Tuple[int, int] = (8, 8)

    # --- Matching ---
    use_cross_check: bool = False      # if True -> BF match + sort (no ratio test)
    ratio_test: float = 0.85           # only used if cross_check=False
    max_matches: int = 1000

    # --- Depth filtering ---
    min_depth_m: float = 0.15
    max_depth_m: float = 4.0

    # --- PnP / RANSAC ---
    pnp_method: int = cv2.SOLVEPNP_EPNP
    ransac_reproj_err_px: float = 10.0
    ransac_confidence: float = 0.999
    ransac_iters: int = 3000
    min_inliers: int = 10

    # --- Refinement ---
    refine: bool = True
    refine_method: int = cv2.SOLVEPNP_ITERATIVE

    # --- Debug ---
    return_debug: bool = True

    # Must be provided by caller:
    # cfg.K_rs = rs_meta["K_color"]
    K_rs: Optional[np.ndarray] = None


def _to_gray(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _valid_uv(u: int, v: int, w: int, h: int) -> bool:
    return (0 <= u < w) and (0 <= v < h)


def backproject_pixel_to_3d(u: float, v: float, Z: float, K: np.ndarray) -> np.ndarray:
    fx = float(K[0, 0]); fy = float(K[1, 1])
    cx = float(K[0, 2]); cy = float(K[1, 2])
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z], dtype=np.float32)


def estimate_pose_brisk(
    etg_bgr: np.ndarray,
    rs_color_bgr: np.ndarray,
    rs_depth_m_aligned: np.ndarray,
    K_etg: np.ndarray,
    dist_etg: Optional[np.ndarray] = None,
    *,
    cfg: BriskPnPConfig = BriskPnPConfig(),
) -> Dict[str, Any]:
    """
    Returns dict with:
        ok, R, t, rvec, tvec, inliers
        stats: num_kp_*, num_good_matches, num_corr, num_inliers, reproj_rmse_px
        debug: keypoints + matches for visualization (optional)
    """
    if cfg.K_rs is None:
        return {"ok": False, "error": "cfg.K_rs not set. Do cfg.K_rs = rs_meta['K_color'] from realsense_stream."}
    K_rs = np.asarray(cfg.K_rs, dtype=np.float32)
    if K_rs.shape != (3, 3):
        return {"ok": False, "error": f"cfg.K_rs must be 3x3, got {K_rs.shape}"}

    K_etg = np.asarray(K_etg, dtype=np.float32)
    if K_etg.shape != (3, 3):
        return {"ok": False, "error": f"K_etg must be 3x3, got {K_etg.shape}"}

    if dist_etg is None:
        dist_etg = np.zeros((5, 1), dtype=np.float32)
    dist_etg = np.asarray(dist_etg, dtype=np.float32).reshape(-1, 1)

    # Depth must be aligned to RS color (same resolution)
    Hc, Wc = rs_color_bgr.shape[:2]
    Hd, Wd = rs_depth_m_aligned.shape[:2]
    if (Hc != Hd) or (Wc != Wd):
        return {"ok": False, "error": f"Depth not aligned to color: color=({Wc}x{Hc}), depth=({Wd}x{Hd})"}

    # --- SCALE MATCHING ---
    # We detect/match features on RS resized to ETG size.
    etg_h, etg_w = etg_bgr.shape[:2]
    scale_x = Wc / float(etg_w)
    scale_y = Hc / float(etg_h)

    rs_color_small = cv2.resize(rs_color_bgr, (etg_w, etg_h), interpolation=cv2.INTER_AREA)

    etg_gray = _to_gray(etg_bgr)
    rs_gray = _to_gray(rs_color_small)

    if cfg.use_clahe:
        clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=cfg.clahe_grid)
        etg_gray = clahe.apply(etg_gray)
        rs_gray = clahe.apply(rs_gray)

    brisk = cv2.BRISK_create(cfg.brisk_thresh, cfg.brisk_octaves, cfg.brisk_pattern_scale)

    kp_etg, des_etg = brisk.detectAndCompute(etg_gray, None)
    kp_rs, des_rs = brisk.detectAndCompute(rs_gray, None)  # kp_rs are in SMALL coords

    if des_etg is None or des_rs is None or len(kp_etg) < 8 or len(kp_rs) < 8:
        return {
            "ok": False,
            "error": "Not enough BRISK features detected.",
            "num_kp_etg": int(0 if kp_etg is None else len(kp_etg)),
            "num_kp_rs_small": int(0 if kp_rs is None else len(kp_rs)),
        }

    # --- MATCH ---
    if cfg.use_cross_check:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_rs, des_etg)  # query=RS(small), train=ETG
        matches = sorted(matches, key=lambda m: m.distance)
        good = matches[: min(cfg.max_matches, len(matches))]
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn = bf.knnMatch(des_rs, des_etg, k=2)
        good = []
        for pair in knn:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < cfg.ratio_test * n.distance:
                good.append(m)
        good = sorted(good, key=lambda m: m.distance)[: min(cfg.max_matches, len(good))]
        
    # --- Homography pre-filter (2D-2D RANSAC) ---
    if len(good) >= 8:
        pts_rs = np.float32([kp_rs[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_etg = np.float32([kp_etg[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
        H, mask = cv2.findHomography(pts_rs, pts_etg, cv2.RANSAC, 5.0)
    
        if mask is not None:
            mask = mask.ravel().astype(bool)
            good = [m for m, keep in zip(good, mask) if keep]



    # Build correspondences: RS 2D (small) -> RS 2D (full) -> depth -> RS 3D; ETG 2D stays as-is.
    obj_pts = []
    img_pts = []
    kept_match_indices = []  # indices into `good` that survived depth filtering (for inlier viz)

    for idx, m in enumerate(good):
        # RS small keypoint
        u_rs_s, v_rs_s = kp_rs[m.queryIdx].pt
        # map to RS full coords
        u_rs = u_rs_s * scale_x
        v_rs = v_rs_s * scale_y

        ui = int(round(u_rs))
        vi = int(round(v_rs))
        if not _valid_uv(ui, vi, Wc, Hc):
            continue

        Z = float(rs_depth_m_aligned[vi, ui])
        if not (cfg.min_depth_m <= Z <= cfg.max_depth_m):
            continue

        # 3D point in RS camera coords (use full-res pixel coords with K_rs)
        P = backproject_pixel_to_3d(u_rs, v_rs, Z, K_rs)

        # ETG point
        u_etg, v_etg = kp_etg[m.trainIdx].pt

        obj_pts.append(P)
        img_pts.append([u_etg, v_etg])
        kept_match_indices.append(idx)

    if len(obj_pts) < cfg.min_inliers:
        return {
            "ok": False,
            "error": f"Not enough valid 2D-3D correspondences: {len(obj_pts)} (good matches={len(good)})",
            "num_kp_etg": int(len(kp_etg)),
            "num_kp_rs_small": int(len(kp_rs)),
            "num_good_matches": int(len(good)),
            "num_corr": int(len(obj_pts)),
        }

    objectPoints = np.asarray(obj_pts, dtype=np.float32).reshape(-1, 1, 3)
    imagePoints = np.asarray(img_pts, dtype=np.float32).reshape(-1, 1, 2)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=objectPoints,
        imagePoints=imagePoints,
        cameraMatrix=K_etg,
        distCoeffs=dist_etg,
        flags=cfg.pnp_method,
        reprojectionError=cfg.ransac_reproj_err_px,
        confidence=cfg.ransac_confidence,
        iterationsCount=cfg.ransac_iters,
    )

    if (not ok) or inliers is None or len(inliers) < cfg.min_inliers:
        return {
            "ok": False,
            "error": f"PnPRansac failed or too few inliers: {0 if inliers is None else len(inliers)}",
            "num_kp_etg": int(len(kp_etg)),
            "num_kp_rs_small": int(len(kp_rs)),
            "num_good_matches": int(len(good)),
            "num_corr": int(len(obj_pts)),
            "num_inliers": int(0 if inliers is None else len(inliers)),
        }

    # Optional refine using inliers only
    if cfg.refine and len(inliers) >= cfg.min_inliers:
        inl_obj = objectPoints[inliers[:, 0]]
        inl_img = imagePoints[inliers[:, 0]]
        try:
            ok2, rvec, tvec = cv2.solvePnP(
                objectPoints=inl_obj,
                imagePoints=inl_img,
                cameraMatrix=K_etg,
                distCoeffs=dist_etg,
                rvec=rvec,
                tvec=tvec,
                useExtrinsicGuess=True,
                flags=cfg.refine_method,
            )
        except Exception:
            pass  # keep original

    R, _ = cv2.Rodrigues(rvec)
    R = R.astype(np.float32)
    t = tvec.astype(np.float32).reshape(3, 1)

    # Reprojection RMSE (inliers)
    reproj_rmse = None
    try:
        inl_obj = objectPoints[inliers[:, 0]]
        inl_img = imagePoints[inliers[:, 0]].reshape(-1, 2)
        proj, _ = cv2.projectPoints(inl_obj, rvec, tvec, K_etg, dist_etg)
        proj = proj.reshape(-1, 2)
        err = proj - inl_img
        reproj_rmse = float(np.sqrt(np.mean(np.sum(err * err, axis=1))))
    except Exception:
        pass

    out: Dict[str, Any] = {
        "ok": True,
        "R": R,
        "t": t,
        "rvec": rvec.astype(np.float32),
        "tvec": tvec.astype(np.float32),
        "inliers": inliers,
        "num_kp_etg": int(len(kp_etg)),
        "num_kp_rs_small": int(len(kp_rs)),
        "num_good_matches": int(len(good)),
        "num_corr": int(len(obj_pts)),
        "num_inliers": int(len(inliers)),
        "reproj_rmse_px": reproj_rmse,
        "scale_rs_to_full": (float(scale_x), float(scale_y)),
    }

    if cfg.return_debug:
        out["debug"] = {
            "kp_etg": kp_etg,
            "kp_rs_small": kp_rs,                  # keypoints are for RS SMALL image
            "rs_color_small": rs_color_small,      # use this for drawMatches
            "matches_good": good,
            "kept_match_indices": kept_match_indices,  # which `good` survived depth filter
        }

    return out


def draw_inlier_matches(
    etg_bgr: np.ndarray,
    rs_color_small_bgr: np.ndarray,
    kp_etg,
    kp_rs_small,
    matches_good,
    kept_match_indices,
    inliers: Optional[np.ndarray],
    max_draw: int = 80,
) -> np.ndarray:
    """
    Draw matches using RS SMALL image (because kp_rs are in that coordinate system).
    We first restrict to matches that survived depth filtering, then restrict to inliers.
    """
    if len(matches_good) == 0:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    # Map inlier indices (which refer to correspondence arrays) -> indices into kept_match_indices -> indices into good
    if inliers is None or len(inliers) == 0:
        chosen_good_idxs = kept_match_indices
    else:
        # inliers[:,0] indexes into objectPoints/imagePoints, i.e., into kept_match_indices order
        chosen_good_idxs = [kept_match_indices[int(i)] for i in inliers[:, 0].tolist()]

    draw_matches = [matches_good[i] for i in chosen_good_idxs]
    draw_matches = draw_matches[:max_draw]

    vis = cv2.drawMatches(
        rs_color_small_bgr, kp_rs_small,
        etg_bgr, kp_etg,
        draw_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return vis
