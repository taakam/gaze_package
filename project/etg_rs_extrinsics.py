#!/usr/bin/env python3
"""
etg_rs_extrinsics.py

Stable estimation of T_etg_in_rs using AprilTags detected in both cameras:
    T_etg_in_rs = T_tag_in_rs * inv(T_tag_in_etg)

Includes:
- smart undistort (fisheye if len=4, otherwise pinhole)
- outlier rejection across per-tag candidates
- temporal smoothing (EMA + quaternion lerp)
- hold last good pose through brief dropouts

NEW:
- configurable small bias transform applied to final T_etg_in_rs:
      T_out = T_bias @ T_etg_in_rs
  where T_bias can be tiny yaw/pitch/roll (and optional translation).
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass
from typing import Dict, Optional, List, Any

import numpy as np
import cv2

try:
    from pupil_apriltags import Detector
except ImportError as e:
    raise SystemExit("pip install pupil-apriltags") from e


# -----------------------------
# SE(3) helpers
# -----------------------------
def Rt_to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = np.asarray(t, dtype=np.float32).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3:4]
    Ti = np.eye(4, dtype=np.float32)
    Ti[:3, :3] = R.T
    Ti[:3, 3:4] = -R.T @ t
    return Ti


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Unit quaternion [w,x,y,z] from rotation matrix."""
    tr = float(np.trace(R))
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float32)
    q /= np.linalg.norm(q) + 1e-12
    return q


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def quat_lerp(q0, q1, a):
    if np.dot(q0, q1) < 0:
        q1 = -q1
    q = (1.0 - a) * q0 + a * q1
    q /= np.linalg.norm(q) + 1e-12
    return q


def smooth_T(T_prev, T_new, a=0.10):
    if T_prev is None:
        return T_new
    q0 = rot_to_quat(T_prev[:3, :3])
    t0 = T_prev[:3, 3].astype(np.float32)

    q1 = rot_to_quat(T_new[:3, :3])
    t1 = T_new[:3, 3].astype(np.float32)

    q = quat_lerp(q0, q1, a)
    t = (1.0 - a) * t0 + a * t1
    return Rt_to_T(quat_to_rot(q), t)


def average_poses(T_list: List[np.ndarray]) -> Optional[np.ndarray]:
    if len(T_list) == 0:
        return None
    ts = np.stack([T[:3, 3] for T in T_list], axis=0)
    t_mean = np.mean(ts, axis=0)

    quats = []
    for T in T_list:
        q = rot_to_quat(T[:3, :3])
        if len(quats) > 0 and np.dot(quats[0], q) < 0:
            q = -q
        quats.append(q)

    q_mean = np.mean(np.stack(quats, axis=0), axis=0)
    q_mean /= np.linalg.norm(q_mean) + 1e-12
    R_mean = quat_to_rot(q_mean)
    return Rt_to_T(R_mean, t_mean)


def average_with_outlier_rejection(T_list: List[np.ndarray], max_dev_m=0.02):
    if len(T_list) == 0:
        return None, []
    ts = np.stack([T[:3, 3] for T in T_list], axis=0)
    t_med = np.median(ts, axis=0)
    kept = [T for T in T_list if np.linalg.norm(T[:3, 3] - t_med) <= max_dev_m]
    if len(kept) == 0:
        return None, []
    return average_poses(kept), kept


def rpy_deg_from_R(R: np.ndarray):
    sy = math.sqrt(float(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]))
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
        pitch = math.atan2(float(-R[2, 0]), sy)
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:
        roll = math.atan2(float(-R[1, 2]), float(R[1, 1]))
        pitch = math.atan2(float(-R[2, 0]), sy)
        yaw = 0.0
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


# -----------------------------
# Bias helper (NEW)
# -----------------------------
def _Rx(a: float) -> np.ndarray:
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=np.float32)

def _Ry(a: float) -> np.ndarray:
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=np.float32)

def _Rz(a: float) -> np.ndarray:
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=np.float32)

def make_bias_T(roll_deg: float, pitch_deg: float, yaw_deg: float,
                tx_m: float = 0.0, ty_m: float = 0.0, tz_m: float = 0.0) -> np.ndarray:
    """
    Bias rotation order: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    This matches the common roll/pitch/yaw convention.
    """
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    R = (_Rz(y) @ _Ry(p) @ _Rx(r)).astype(np.float32)
    t = np.array([tx_m, ty_m, tz_m], dtype=np.float32)
    return Rt_to_T(R, t)


# -----------------------------
# Undistortion (smart)
# -----------------------------
def undistort_with_K(gray: np.ndarray, K: np.ndarray, dist: np.ndarray, *, fisheye_balance: float = 0.3):
    """
    Returns: (und, K_use)
    dist len 4 => OpenCV fisheye model
    else => standard pinhole (radtan) model
    """
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    h, w = gray.shape[:2]

    if dist.size == 4:
        D = dist.reshape(4, 1)
        Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (w, h), np.eye(3), balance=fisheye_balance
        )
        und = cv2.fisheye.undistortImage(gray, K, D, Knew=Knew)
        return und, Knew.astype(np.float32)

    D = dist.reshape(-1, 1)
    Knew, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0.0, newImgSize=(w, h))
    und = cv2.undistort(gray, K, D, None, Knew)
    return und, Knew.astype(np.float32)


# -----------------------------
# Config + Estimator
# -----------------------------
@dataclass
class ExtrinsicsConfig:
    tag_family: str = "tag36h11"
    tag_size_m: float = 0.162

    min_decision_margin: float = 25.0
    fisheye_balance: float = 0.3

    outlier_trans_thresh_m: float = 0.02
    smooth_alpha: float = 0.10
    hold_seconds: float = 0.7

    # detector tuning
    nthreads: int = 4
    quad_decimate: float = 1.0
    quad_sigma: float = 0.0
    refine_edges: int = 1
    decode_sharpening: float = 0.25

    # ---- NEW: bias correction ----
    # Start with yaw only, e.g. +0.5 or -0.5 deg, then tune.
    bias_roll_deg: float = 0.0
    bias_pitch_deg: float = 0.0
    bias_yaw_deg: float = 0.0

    # optional translation bias (usually keep at 0)
    bias_tx_m: float = 0.0
    bias_ty_m: float = 0.0
    bias_tz_m: float = 0.0


class ETGRSExtrinsicsEstimator:
    def __init__(self, cfg: ExtrinsicsConfig):
        self.cfg = cfg
        self.detector = Detector(
            families=cfg.tag_family,
            nthreads=cfg.nthreads,
            quad_decimate=cfg.quad_decimate,
            quad_sigma=cfg.quad_sigma,
            refine_edges=cfg.refine_edges,
            decode_sharpening=cfg.decode_sharpening,
        )
        self._last_good_T: Optional[np.ndarray] = None
        self._last_good_time: float = 0.0
        self._T_smooth: Optional[np.ndarray] = None

        # precompute bias transform
        self._T_bias = make_bias_T(
            cfg.bias_roll_deg, cfg.bias_pitch_deg, cfg.bias_yaw_deg,
            cfg.bias_tx_m, cfg.bias_ty_m, cfg.bias_tz_m
        ).astype(np.float32)

    def _detect_tag_poses(self, gray: np.ndarray, K: np.ndarray, dist: np.ndarray):
        und, K_use = undistort_with_K(gray, K, dist, fisheye_balance=self.cfg.fisheye_balance)

        fx, fy = float(K_use[0, 0]), float(K_use[1, 1])
        cx, cy = float(K_use[0, 2]), float(K_use[1, 2])

        detections = self.detector.detect(
            und,
            estimate_tag_pose=True,
            camera_params=(fx, fy, cx, cy),
            tag_size=self.cfg.tag_size_m,
        )

        out: Dict[int, np.ndarray] = {}
        for d in detections:
            if d.decision_margin < self.cfg.min_decision_margin:
                continue
            R = d.pose_R.astype(np.float32)
            t = d.pose_t.astype(np.float32).reshape(3, 1)
            out[int(d.tag_id)] = Rt_to_T(R, t)

        return out, und, detections

    def update(
        self,
        etg_bgr: np.ndarray,
        rs_bgr: np.ndarray,
        K_etg: np.ndarray,
        dist_etg: np.ndarray,
        K_rs: np.ndarray,
        dist_rs: np.ndarray,
        *,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Returns dict:
          ok: bool
          T_etg_in_rs: 4x4 (if ok)
          common_ids, kept, reason, rpy_deg, t_m
          vis_etg, vis_rs
          dets_etg, dets_rs
        """
        if now is None:
            now = time.time()

        etg_gray = cv2.cvtColor(etg_bgr, cv2.COLOR_BGR2GRAY)
        rs_gray = cv2.cvtColor(rs_bgr, cv2.COLOR_BGR2GRAY)

        tags_etg, etg_used, dets_etg = self._detect_tag_poses(etg_gray, K_etg, dist_etg)
        tags_rs, rs_used, dets_rs = self._detect_tag_poses(rs_gray, K_rs, dist_rs)

        common = sorted(set(tags_etg.keys()) & set(tags_rs.keys()))

        valid = False
        T_etg_in_rs = None
        kept_count = 0
        reason = ""

        if len(common) > 0:
            candidates = []
            for tid in common:
                T_tag_in_rs = tags_rs[tid]
                T_tag_in_etg = tags_etg[tid]
                candidates.append(T_tag_in_rs @ inv_T(T_tag_in_etg))

            T_avg, kept = average_with_outlier_rejection(
                candidates, max_dev_m=self.cfg.outlier_trans_thresh_m
            )
            kept_count = len(kept)

            if T_avg is not None:
                T_etg_in_rs = T_avg
                valid = True
            else:
                reason = "all candidates rejected as outliers"
        else:
            reason = "no common tags"

        # hold last good
        if valid:
            self._last_good_T = T_etg_in_rs
            self._last_good_time = now
        else:
            if self._last_good_T is not None and (now - self._last_good_time) <= self.cfg.hold_seconds:
                T_etg_in_rs = self._last_good_T
                valid = True
                reason = "held last good"

        # smooth
        if valid and T_etg_in_rs is not None:
            self._T_smooth = smooth_T(self._T_smooth, T_etg_in_rs, a=self.cfg.smooth_alpha)
            T_etg_in_rs = self._T_smooth

        # ---- NEW: apply bias correction at the very end ----
        if valid and T_etg_in_rs is not None:
            T_etg_in_rs = (self._T_bias @ T_etg_in_rs).astype(np.float32)

        out: Dict[str, Any] = {
            "ok": bool(valid and (T_etg_in_rs is not None)),
            "T_etg_in_rs": T_etg_in_rs,
            "common_ids": common,
            "kept": kept_count,
            "reason": reason,
            "vis_etg": etg_used,
            "vis_rs": rs_used,
            "dets_etg": dets_etg,
            "dets_rs": dets_rs,
            "num_etg": len(tags_etg),
            "num_rs": len(tags_rs),
        }

        if out["ok"]:
            t = T_etg_in_rs[:3, 3]
            rpy = rpy_deg_from_R(T_etg_in_rs[:3, :3])
            out["t_m"] = t
            out["rpy_deg"] = rpy

            # helpful debug: show bias too
            out["bias_rpy_deg"] = (self.cfg.bias_roll_deg, self.cfg.bias_pitch_deg, self.cfg.bias_yaw_deg)
            out["bias_t_m"] = (self.cfg.bias_tx_m, self.cfg.bias_ty_m, self.cfg.bias_tz_m)

        return out