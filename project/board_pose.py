#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import cv2

try:
    from pupil_apriltags import Detector
except ImportError as e:
    raise SystemExit("pip install pupil-apriltags") from e


def Rt_to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = np.asarray(t, dtype=np.float32).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float32)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


@dataclass
class BoardConfig:
    tag_family: str = "tag36h11"
    tag_size_m: float = 0.162
    horizontal_inner_gap_m: float = 0.306
    vertical_inner_gap_m: float = 0.024

    # map your actual tag IDs here
    tag_ids: Tuple[int, int, int, int] = (0, 1, 2, 3)  # TL, TR, BL, BR

    min_decision_margin: float = 25.0
    nthreads: int = 4
    quad_decimate: float = 1.0
    quad_sigma: float = 0.0
    refine_edges: int = 1
    decode_sharpening: float = 0.25


class AprilTagBoardPoseEstimator:
    """
    Estimates T_board_in_cam from a 2x2 AprilTag board using solvePnP.
    board_frame:
      origin = center of the 4 tag centers
      +x = right
      +y = up
      +z = out of board plane
    """

    def __init__(self, cfg: BoardConfig):
        self.cfg = cfg
        self.detector = Detector(
            families=cfg.tag_family,
            nthreads=cfg.nthreads,
            quad_decimate=cfg.quad_decimate,
            quad_sigma=cfg.quad_sigma,
            refine_edges=cfg.refine_edges,
            decode_sharpening=cfg.decode_sharpening,
        )
        self._tag_centers = self._make_tag_centers()

    def _make_tag_centers(self) -> Dict[int, np.ndarray]:
        s = self.cfg.tag_size_m
        dx = 0.5 * (s + self.cfg.horizontal_inner_gap_m)  # 0.234
        dy = 0.5 * (s + self.cfg.vertical_inner_gap_m)    # 0.093

        tl, tr, bl, br = self.cfg.tag_ids
        return {
            tl: np.array([-dx, +dy, 0.0], dtype=np.float32),
            tr: np.array([+dx, +dy, 0.0], dtype=np.float32),
            bl: np.array([-dx, -dy, 0.0], dtype=np.float32),
            br: np.array([+dx, -dy, 0.0], dtype=np.float32),
        }

    def _tag_object_corners(self, tag_id: int) -> np.ndarray:
        """
        Returns 4x3 object points for one tag, in the same corner order
        as pupil_apriltags detection.corners:
            [top-left, top-right, bottom-right, bottom-left]
        """
        c = self._tag_centers[tag_id]
        h = self.cfg.tag_size_m / 2.0

        # x right, y up on board
        # image-style top-left corresponds to +y and -x in board coords
        pts = np.array([
            [c[0] - h, c[1] + h, 0.0],  # top-left
            [c[0] + h, c[1] + h, 0.0],  # top-right
            [c[0] + h, c[1] - h, 0.0],  # bottom-right
            [c[0] - h, c[1] - h, 0.0],  # bottom-left
        ], dtype=np.float32)
        return pts

    def estimate(
        self,
        bgr: np.ndarray,
        K: np.ndarray,
        dist: np.ndarray,
    ) -> Dict[str, Any]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        detections = self.detector.detect(gray, estimate_tag_pose=False)

        obj_pts: List[np.ndarray] = []
        img_pts: List[np.ndarray] = []
        used_ids: List[int] = []

        for d in detections:
            tid = int(d.tag_id)
            if tid not in self._tag_centers:
                continue
            if float(d.decision_margin) < self.cfg.min_decision_margin:
                continue

            obj_pts.append(self._tag_object_corners(tid))
            img_pts.append(np.asarray(d.corners, dtype=np.float32))
            used_ids.append(tid)

        if len(obj_pts) == 0:
            return {"ok": False, "reason": "no board tags found", "used_ids": []}

        obj = np.concatenate(obj_pts, axis=0).astype(np.float32)   # Nx3
        img = np.concatenate(img_pts, axis=0).astype(np.float32)   # Nx2

        K = np.asarray(K, dtype=np.float32).reshape(3, 3)
        dist = np.asarray(dist, dtype=np.float32).reshape(-1, 1)

        ok, rvec, tvec = cv2.solvePnP(
            obj,
            img,
            K,
            dist,
            flags=cv2.SOLVEPNP_IPPE if len(obj) >= 4 else cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok:
            return {"ok": False, "reason": "solvePnP failed", "used_ids": used_ids}

        R, _ = cv2.Rodrigues(rvec)
        T_board_in_cam = Rt_to_T(R, tvec.reshape(3))

        return {
            "ok": True,
            "T_board_in_cam": T_board_in_cam,
            "used_ids": sorted(set(used_ids)),
            "num_corners": int(obj.shape[0]),
            "rvec": rvec,
            "tvec": tvec,
        }