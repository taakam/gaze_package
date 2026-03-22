#!/usr/bin/env python3
"""
compute_T_etg_in_rs_live.py  (FULL UPDATED + STABLE POSE)

Adds:
- Smart undistort (ETG fisheye len=4, RS standard len=5)
- Outlier rejection across tag-derived candidate transforms
- EMA smoothing on translation + quaternion-lerp smoothing on rotation
- Holds last good pose through brief dropouts

Per-tag:
    T_etg_in_rs = T_tag_in_rs * inv(T_tag_in_etg)

Then:
- reject outliers
- average remaining
- smooth over time

Run:
    python3 compute_T_etg_in_rs_live.py
"""

import time
import math
import numpy as np
import cv2
import msgpack

from pupil_stream import PupilStream, PupilConfig
from realsense_stream import RealSenseStream, RealSenseConfig

try:
    from pupil_apriltags import Detector
except ImportError as e:
    raise SystemExit("pip install pupil-apriltags") from e


# -----------------------------
# USER CONFIG
# -----------------------------
TAG_FAMILY = "tag36h11"
TAG_SIZE_M = 0.09               # black square size (meters) - MEASURE THIS
MIN_DECISION_MARGIN = 25.0      # stricter -> less jitter, fewer detections
HOLD_SECONDS = 0.7              # keep last good pose through brief dropouts

# Undistort options
FISHEYE_BALANCE = 0.3           # 0.0 crops more, 0.5 keeps more FOV

# Multi-tag averaging / rejection
OUTLIER_TRANS_THRESH_M = 0.02   # reject candidate transforms >2cm from median translation

# Temporal smoothing
SMOOTH_ALPHA = 0.10             # 0.05 smoother, 0.2 more responsive


# -----------------------------
# Intrinsics loader (ETG)
# -----------------------------
def load_world_intrinsics(path: str = "world.intrinsics"):
    with open(path, "rb") as fh:
        data = msgpack.unpack(fh, raw=False)

    res_key = next(k for k in data.keys() if k != "version")
    block = data[res_key]

    K = np.array(block["camera_matrix"], dtype=np.float32).reshape(3, 3)

    if "dist_coefs" in block:
        dist = np.array(block["dist_coefs"], dtype=np.float32).reshape(-1, 1)
    elif "dist_coeffs" in block:
        dist = np.array(block["dist_coeffs"], dtype=np.float32).reshape(-1, 1)
    else:
        dist = np.zeros((4, 1), dtype=np.float32)

    print(f"[ETG] intrinsics key: {res_key} ({type(res_key)})")
    print("[ETG] K:\n", K)
    print("[ETG] dist (len={}): {}".format(dist.reshape(-1).size, dist.reshape(-1)))
    return K, dist


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
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ], dtype=np.float32)


def average_poses(T_list):
    """Mean translation + quaternion average rotation."""
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


def average_with_outlier_rejection(T_list, max_dev_m=0.02):
    """
    Reject transforms whose translation deviates from median by > max_dev_m.
    Returns (T_avg, kept_list)
    """
    if len(T_list) == 0:
        return None, []
    ts = np.stack([T[:3, 3] for T in T_list], axis=0)
    t_med = np.median(ts, axis=0)

    kept = []
    for T in T_list:
        if np.linalg.norm(T[:3, 3] - t_med) <= max_dev_m:
            kept.append(T)

    if len(kept) == 0:
        return None, []
    return average_poses(kept), kept


def rpy_deg_from_R(R: np.ndarray):
    sy = math.sqrt(float(R[0, 0]*R[0, 0] + R[1, 0]*R[1, 0]))
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


# -----------------------------
# Undistortion (smart)
# -----------------------------
def undistort_with_K(gray: np.ndarray, K: np.ndarray, dist: np.ndarray):
    """
    Returns: (und, K_use)
    - dist len 4 => fisheye
    - else => standard OpenCV
    """
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    h, w = gray.shape[:2]

    if dist.size == 4:
        D = dist.reshape(4, 1)
        Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (w, h), np.eye(3), balance=FISHEYE_BALANCE
        )
        und = cv2.fisheye.undistortImage(gray, K, D, Knew=Knew)
        return und, Knew.astype(np.float32)

    D = dist.reshape(-1, 1)
    Knew, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0.0, newImgSize=(w, h))
    und = cv2.undistort(gray, K, D, None, Knew)
    return und, Knew.astype(np.float32)


# -----------------------------
# AprilTag detection
# -----------------------------
def detect_tag_poses(detector: Detector, gray: np.ndarray, K: np.ndarray, dist: np.ndarray):
    und, K_use = undistort_with_K(gray, K, dist)

    fx, fy = float(K_use[0, 0]), float(K_use[1, 1])
    cx, cy = float(K_use[0, 2]), float(K_use[1, 2])

    detections = detector.detect(
        und,
        estimate_tag_pose=True,
        camera_params=(fx, fy, cx, cy),
        tag_size=TAG_SIZE_M,
    )

    out = {}
    for d in detections:
        if d.decision_margin < MIN_DECISION_MARGIN:
            continue
        R = d.pose_R.astype(np.float32)
        t = d.pose_t.astype(np.float32).reshape(3, 1)
        out[int(d.tag_id)] = Rt_to_T(R, t)

    return out, und, detections


# -----------------------------
# Main
# -----------------------------
def main():
    K_etg, dist_etg = load_world_intrinsics("world.intrinsics")

    rs_cfg = RealSenseConfig(
        color_wh_fps=(1280, 720, 30),
        depth_wh_fps=(848, 480, 30),
        align_to_color=True,
        enable_filters=False,
        max_depth_m=4.0,
    )
    pupil_cfg = PupilConfig(topic="frame.world", format="bgr", buffer_size=1)

    detector = Detector(
        families=TAG_FAMILY,
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
    )

    cv2.namedWindow("ETG (used for detection)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("RS (used for detection)", cv2.WINDOW_NORMAL)

    last_good_T = None
    last_good_time = 0.0
    T_smooth = None
    printed_rs = False

    with PupilStream(pupil_cfg) as pupil, RealSenseStream(rs_cfg) as rs_cam:
        while True:
            etg_bgr, _ = pupil.read()
            rs_bgr, _, rs_meta = rs_cam.read()

            # RS intrinsics
            K_rs = np.asarray(rs_meta["K_color"], dtype=np.float32).reshape(3, 3)
            dist_rs = np.asarray(rs_meta.get("dist_color", np.zeros((5, 1))), dtype=np.float32).reshape(-1, 1)

            if not printed_rs:
                print("[RS ] K:\n", K_rs)
                print("[RS ] dist (len={}): {}".format(dist_rs.reshape(-1).size, dist_rs.reshape(-1)))
                printed_rs = True

            etg_gray = cv2.cvtColor(etg_bgr, cv2.COLOR_BGR2GRAY)
            rs_gray = cv2.cvtColor(rs_bgr, cv2.COLOR_BGR2GRAY)

            tags_etg, etg_used, dets_etg = detect_tag_poses(detector, etg_gray, K_etg, dist_etg)
            tags_rs, rs_used, dets_rs = detect_tag_poses(detector, rs_gray, K_rs, dist_rs)

            common = sorted(set(tags_etg.keys()) & set(tags_rs.keys()))
            now = time.time()

            valid = False
            T_etg_in_rs = None
            kept_count = 0

            if len(common) > 0:
                candidates = []
                for tid in common:
                    T_tag_in_rs = tags_rs[tid]
                    T_tag_in_etg = tags_etg[tid]
                    candidates.append(T_tag_in_rs @ inv_T(T_tag_in_etg))

                T_avg, kept = average_with_outlier_rejection(
                    candidates, max_dev_m=OUTLIER_TRANS_THRESH_M
                )
                kept_count = len(kept)

                if T_avg is not None:
                    T_etg_in_rs = T_avg
                    valid = True

            # Hold last good through brief dropouts
            if valid:
                last_good_T = T_etg_in_rs
                last_good_time = now
            else:
                if last_good_T is not None and (now - last_good_time) <= HOLD_SECONDS:
                    T_etg_in_rs = last_good_T
                    valid = True

            # Smooth if we have something valid
            if valid and T_etg_in_rs is not None:
                T_smooth = smooth_T(T_smooth, T_etg_in_rs, a=SMOOTH_ALPHA)
                T_etg_in_rs = T_smooth

            # Visualize detections
            etg_vis = cv2.cvtColor(etg_used, cv2.COLOR_GRAY2BGR)
            rs_vis = cv2.cvtColor(rs_used, cv2.COLOR_GRAY2BGR)

            for d in dets_etg:
                if d.decision_margin < MIN_DECISION_MARGIN:
                    continue
                c = tuple(map(int, d.center))
                cv2.circle(etg_vis, c, 4, (0, 255, 0), -1)
                cv2.putText(etg_vis, f"id={d.tag_id}", (c[0] + 6, c[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            for d in dets_rs:
                if d.decision_margin < MIN_DECISION_MARGIN:
                    continue
                c = tuple(map(int, d.center))
                cv2.circle(rs_vis, c, 4, (0, 255, 0), -1)
                cv2.putText(rs_vis, f"id={d.tag_id}", (c[0] + 6, c[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if valid and T_etg_in_rs is not None:
                t = T_etg_in_rs[:3, 3]
                rpy = rpy_deg_from_R(T_etg_in_rs[:3, :3])

                txt = (
                    f"T_etg_in_rs: t=[{t[0]:+.3f},{t[1]:+.3f},{t[2]:+.3f}] m  "
                    f"rpy=[{rpy[0]:+.1f},{rpy[1]:+.1f},{rpy[2]:+.1f}] deg  "
                    f"common={common} kept={kept_count}"
                )
                cv2.putText(rs_vis, "VALID", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(rs_vis, txt, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # print at ~2 Hz
                if int(now * 2) != int((now - 0.5) * 2):
                    print(txt)
            else:
                cv2.putText(rs_vis, f"NO COMMON TAGS (ETG:{len(tags_etg)} RS:{len(tags_rs)})",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("ETG (used for detection)", etg_vis)
            cv2.imshow("RS (used for detection)", rs_vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()