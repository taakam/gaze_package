#!/usr/bin/env python3
"""
gaze_raytrace.py

Given:
- gaze pixel in ETG image (u,v)
- ETG intrinsics + distortion
- T_etg_in_rs (ETG -> RS)
- RS intrinsics
- RS depth aligned to color (meters)

Compute:
- 3D fixation point in RS frame by ray marching + depth surface intersection.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import cv2


@dataclass
class RaytraceConfig:
    # ray marching limits in meters
    s_min: float = 0.10
    s_max: float = 3.50

    # coarse-to-fine sampling
    coarse_steps: int = 80
    refine_steps: int = 20

    # hit condition: |z_ray - z_depth| < eps
    z_epsilon_m: float = 0.02

    # ignore invalid depths
    min_depth_m: float = 0.15
    max_depth_m: float = 4.0

    # if depth missing at pixel, search small neighborhood
    neighborhood: int = 2  # radius in pixels


def _undistort_pixel_to_unit_ray(u: float, v: float, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """
    Uses cv2.undistortPoints to get normalized (x,y) then returns a 3D direction [x,y,1] normalized.
    Assumes standard pinhole distortion (radtan). Works with len=4/5/8 etc, as long as OpenCV undistortPoints supports it.
    """
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)

    pts = np.array([[[u, v]]], dtype=np.float64)  # shape (1,1,2)
    und = cv2.undistortPoints(pts, K, dist, P=None)  # returns normalized coords
    x, y = float(und[0, 0, 0]), float(und[0, 0, 1])

    d = np.array([x, y, 1.0], dtype=np.float32)
    d /= (np.linalg.norm(d) + 1e-12)
    return d


def _project_rs(X_rs: np.ndarray, K_rs: np.ndarray) -> Optional[Tuple[float, float]]:
    x, y, z = float(X_rs[0]), float(X_rs[1]), float(X_rs[2])
    if z <= 1e-6:
        return None
    fx, fy = float(K_rs[0, 0]), float(K_rs[1, 1])
    cx, cy = float(K_rs[0, 2]), float(K_rs[1, 2])
    u = fx * x / z + cx
    v = fy * y / z + cy
    return u, v


def _depth_at(depth_m: np.ndarray, u: float, v: float, cfg: RaytraceConfig) -> Optional[float]:
    h, w = depth_m.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    r = int(cfg.neighborhood)

    best = None
    best_abs = 1e9

    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            x = ui + dx
            y = vi + dy
            if x < 0 or x >= w or y < 0 or y >= h:
                continue
            z = float(depth_m[y, x])
            if not (cfg.min_depth_m <= z <= cfg.max_depth_m):
                continue
            # prefer the closest pixel to requested one
            score = abs(dx) + abs(dy)
            if score < best_abs:
                best_abs = score
                best = z
    return best


def raytrace_fixation_rs(
    gaze_uv_etg: Tuple[float, float],
    K_etg: np.ndarray,
    dist_etg: np.ndarray,
    T_etg_in_rs: np.ndarray,
    K_rs: np.ndarray,
    depth_rs_aligned_m: np.ndarray,
    *,
    cfg: RaytraceConfig = RaytraceConfig(),
) -> Dict[str, Any]:
    """
    Returns:
      ok, X_rs (3,), uv_rs (2,), s_hit, debug
    """
    u, v = float(gaze_uv_etg[0]), float(gaze_uv_etg[1])

    # 1) ETG ray
    d_etg = _undistort_pixel_to_unit_ray(u, v, K_etg, dist_etg)
    o_etg = np.zeros((3,), dtype=np.float32)

    # 2) transform ray to RS
    R = T_etg_in_rs[:3, :3].astype(np.float32)
    t = T_etg_in_rs[:3, 3].astype(np.float32)

    o_rs = t
    d_rs = (R @ d_etg.reshape(3, 1)).reshape(3)
    d_rs /= (np.linalg.norm(d_rs) + 1e-12)

    # 3) march along ray (coarse)
    s_vals = np.linspace(cfg.s_min, cfg.s_max, cfg.coarse_steps, dtype=np.float32)

    best = None
    for s in s_vals:
        X = o_rs + s * d_rs
        uv = _project_rs(X, K_rs)
        if uv is None:
            continue
        u_rs, v_rs = uv
        z_meas = _depth_at(depth_rs_aligned_m, u_rs, v_rs, cfg)
        if z_meas is None:
            continue

        z_ray = float(X[2])
        if abs(z_ray - z_meas) < cfg.z_epsilon_m:
            best = (float(s), X, (u_rs, v_rs), z_meas, z_ray)
            break

    if best is None:
        return {"ok": False, "error": "no depth intersection found", "o_rs": o_rs, "d_rs": d_rs}

    # 4) refine around hit
    s0 = best[0]
    s_lo = max(cfg.s_min, s0 - (cfg.s_max - cfg.s_min) / cfg.coarse_steps)
    s_hi = min(cfg.s_max, s0 + (cfg.s_max - cfg.s_min) / cfg.coarse_steps)

    s_vals2 = np.linspace(s_lo, s_hi, cfg.refine_steps, dtype=np.float32)
    best2 = None
    best_err = 1e9

    for s in s_vals2:
        X = o_rs + s * d_rs
        uv = _project_rs(X, K_rs)
        if uv is None:
            continue
        u_rs, v_rs = uv
        z_meas = _depth_at(depth_rs_aligned_m, u_rs, v_rs, cfg)
        if z_meas is None:
            continue
        err = abs(float(X[2]) - z_meas)
        if err < best_err:
            best_err = err
            best2 = (float(s), X, (u_rs, v_rs), z_meas, float(X[2]))

    if best2 is None:
        s_hit, X_hit, uv_hit, z_meas, z_ray = best
    else:
        s_hit, X_hit, uv_hit, z_meas, z_ray = best2

    return {
        "ok": True,
        "X_rs": np.asarray(X_hit, dtype=np.float32),
        "uv_rs": (float(uv_hit[0]), float(uv_hit[1])),
        "s_hit": float(s_hit),
        "depth_meas_m": float(z_meas),
        "depth_ray_m": float(z_ray),
        "depth_err_m": float(abs(z_ray - z_meas)),
        "o_rs": o_rs,
        "d_rs": d_rs,
    }