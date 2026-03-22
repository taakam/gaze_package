#!/usr/bin/env python3
"""
run_fixation_3d_live.py  (FULL UPDATED)

What this does (live):
1) Reads ETG world frame (Pupil) + RealSense aligned color+depth
2) Estimates / updates ETG->RS extrinsics using AprilTags (your etg_rs_extrinsics module)
3) Subscribes to Pupil Network API:
     - pupil.0.2d
     - pupil.1.2d
     - gaze (and gaze.* variants)
     - fixation (optional)
4) Converts gaze norm_pos -> ETG pixel (IMPORTANT: Y is flipped in Pupil)
5) Undistorts that pixel to a unit ray in ETG cam coords (fisheye vs pinhole handled)
6) Transforms ray into RS cam coords using T_etg_in_rs
7) Ray-marches through RS depth to find a 3D hit point
8) Visualizes:
     - ETG window with gaze dot
     - RS window with hit dot + XYZ text + extrinsics status

Run:
    python3 run_fixation_3d_live.py

Requirements:
    pip install pyzmq msgpack pupil-apriltags opencv-python numpy
and your existing modules:
    - realsense_stream.py
    - pupil_stream.py
    - etg_rs_extrinsics.py   (the estimator you pasted)
"""

import time
from typing import Optional, Tuple, Dict, Any

import numpy as np
import cv2
import zmq
import msgpack

from realsense_stream import RealSenseStream, RealSenseConfig
from pupil_stream import PupilStream, PupilConfig

from etg_rs_extrinsics import ETGRSExtrinsicsEstimator, ExtrinsicsConfig


# -----------------------------
# USER CONFIG
# -----------------------------
PUPIL_REMOTE_IP = "127.0.0.1"
PUPIL_REMOTE_PORT = 50020

ETG_W = 640
ETG_H = 480

MIN_DEPTH_M = 0.15
MAX_DEPTH_M = 4.0

# Ray marching params (tweak)
RAY_STEPS = 180
RAY_STEP_M = 0.015       # 1.5 cm step
HIT_THRESH_M = 0.025     # 2.5 cm threshold
NEIGHBORHOOD = 1         # check +/-1 pixel neighborhood for depth match

SHOW_WINDOWS = True


# -----------------------------
# Pupil Network API (ZMQ) helpers
# -----------------------------
def _req(remote: zmq.Socket, s: str) -> str:
    remote.send_string(s)
    return remote.recv_string()


class PupilSubscriber:
    def __init__(self, ip=PUPIL_REMOTE_IP, port=PUPIL_REMOTE_PORT, topics=()):
        self.ip = ip
        self.port = port
        self.topics = list(topics)
        self.ctx = None
        self.remote = None
        self.sub = None

    def start(self):
        self.ctx = zmq.Context.instance()

        self.remote = self.ctx.socket(zmq.REQ)
        self.remote.connect(f"tcp://{self.ip}:{self.port}")

        sub_port = _req(self.remote, "SUB_PORT")

        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(f"tcp://{self.ip}:{sub_port}")

        for t in self.topics:
            self.sub.setsockopt_string(zmq.SUBSCRIBE, t)

        return self

    def recv(self, timeout_ms: int = 0) -> Optional[Tuple[str, Dict[str, Any]]]:
        if self.sub is None:
            raise RuntimeError("Subscriber not started.")

        if timeout_ms > 0:
            poller = zmq.Poller()
            poller.register(self.sub, zmq.POLLIN)
            evts = dict(poller.poll(timeout_ms))
            if self.sub not in evts:
                return None

        try:
            topic_b, payload = self.sub.recv_multipart(flags=zmq.NOBLOCK if timeout_ms == 0 else 0)
        except zmq.Again:
            return None

        topic = topic_b.decode("utf-8", errors="replace")
        msg = msgpack.loads(payload, raw=False)
        return topic, msg

    def stop(self):
        if self.sub is not None:
            self.sub.close(0)
        if self.remote is not None:
            self.remote.close(0)
        self.sub = None
        self.remote = None


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
    print("[ETG] dist len:", dist.reshape(-1).size, " vals:", dist.reshape(-1))
    return K, dist


# -----------------------------
# Gaze conversion helpers
# -----------------------------
def pupil_norm_to_pixel(norm_pos: Tuple[float, float], W: int, H: int) -> Tuple[float, float]:
    """
    IMPORTANT:
      Pupil norm_pos is normalized [0..1] with ORIGIN AT BOTTOM-LEFT.
      OpenCV pixels assume origin top-left.
    """
    x, y = float(norm_pos[0]), float(norm_pos[1])
    x = 0.0 if x < 0 else (1.0 if x > 1 else x)
    y = 0.0 if y < 0 else (1.0 if y > 1 else y)
    u = x * (W - 1)
    v = (1.0 - y) * (H - 1)  # <-- flip Y
    return u, v


def gaze_pixel_to_unit_ray_etg(u: float, v: float, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """
    Compute a unit bearing ray in ETG camera coordinates from a pixel, accounting for distortion.
    - If dist has len=4 => fisheye model
    - else => standard radtan
    """
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    d = np.asarray(dist, dtype=np.float64).reshape(-1)

    pts = np.array([[[u, v]]], dtype=np.float64)  # (1,1,2)

    if d.size == 4:
        D = d.reshape(4, 1)
        und = cv2.fisheye.undistortPoints(pts, K, D)  # normalized coords (x,y)
        x = float(und[0, 0, 0])
        y = float(und[0, 0, 1])
    else:
        D = d.reshape(-1, 1)
        und = cv2.undistortPoints(pts, K, D)
        x = float(und[0, 0, 0])
        y = float(und[0, 0, 1])

    ray = np.array([x, y, 1.0], dtype=np.float32)
    ray /= (np.linalg.norm(ray) + 1e-12)
    return ray


def transform_ray_etg_to_rs(T_etg_in_rs: np.ndarray, d_etg_unit: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given ETG->RS transform and a unit direction in ETG camera coords,
    return (origin_rs, dir_rs_unit) in RS camera coords.
    """
    R = T_etg_in_rs[:3, :3].astype(np.float32)
    t = T_etg_in_rs[:3, 3].astype(np.float32)
    origin_rs = t.copy()
    dir_rs = (R @ d_etg_unit.reshape(3, 1)).reshape(3).astype(np.float32)
    dir_rs /= (np.linalg.norm(dir_rs) + 1e-12)
    return origin_rs, dir_rs


def project_rs_to_pixel(P_rs: np.ndarray, K_rs: np.ndarray) -> Tuple[int, int]:
    X, Y, Z = float(P_rs[0]), float(P_rs[1]), float(P_rs[2])
    fx, fy = float(K_rs[0, 0]), float(K_rs[1, 1])
    cx, cy = float(K_rs[0, 2]), float(K_rs[1, 2])
    u = int(round(fx * X / Z + cx))
    v = int(round(fy * Y / Z + cy))
    return u, v


def raycast_depth(origin_rs: np.ndarray, dir_rs: np.ndarray, depth_m: np.ndarray, K_rs: np.ndarray) -> Optional[np.ndarray]:
    """
    Ray march:
    - sample points along ray
    - project to pixel
    - compare ray depth vs measured depth (optionally in small neighborhood)
    """
    H, W = depth_m.shape[:2]

    for i in range(1, RAY_STEPS + 1):
        P = origin_rs + dir_rs * (i * RAY_STEP_M)
        Z = float(P[2])
        if Z <= 0 or Z < MIN_DEPTH_M or Z > MAX_DEPTH_M:
            continue

        u, v = project_rs_to_pixel(P, K_rs)
        if not (0 <= u < W and 0 <= v < H):
            continue

        best = None
        for dv in range(-NEIGHBORHOOD, NEIGHBORHOOD + 1):
            for du in range(-NEIGHBORHOOD, NEIGHBORHOOD + 1):
                uu, vv = u + du, v + dv
                if 0 <= uu < W and 0 <= vv < H:
                    z_meas = float(depth_m[vv, uu])
                    if z_meas > 0 and MIN_DEPTH_M <= z_meas <= MAX_DEPTH_M:
                        err = abs(Z - z_meas)
                        if best is None or err < best:
                            best = err

        if best is not None and best < HIT_THRESH_M:
            return P.astype(np.float32)

    return None


# -----------------------------
# Pupil payload parsing helpers
# -----------------------------
def extract_gaze_norm_pos(topic: str, payload: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """
    Tries to pull a usable 2D gaze coordinate from common Pupil topics.
    We only accept things that represent gaze on the world image.

    Typical:
      topic == "gaze"  with payload["norm_pos"] = [x,y]
      topic like "gaze.2d.0" or "gaze.2d" similarly
      fixation also has norm_pos sometimes
    """
    if "norm_pos" in payload:
        try:
            a = payload["norm_pos"]
            return (float(a[0]), float(a[1]))
        except Exception:
            return None
    return None


# -----------------------------
# Main
# -----------------------------
def main():
    # ETG intrinsics (distortion needed for correct gaze ray)
    K_etg, dist_etg = load_world_intrinsics("world.intrinsics")

    # RealSense
    rs_cfg = RealSenseConfig(
        color_wh_fps=(1280, 720, 30),
        depth_wh_fps=(848, 480, 30),
        align_to_color=True,
        enable_filters=False,
        max_depth_m=MAX_DEPTH_M,
    )

    # Pupil world frame via your PupilStream (keeps ETG image in sync)
    pupil_cfg = PupilConfig(topic="frame.world", format="bgr", buffer_size=1)

    # AprilTag extrinsics estimator
    ext_cfg = ExtrinsicsConfig(
        tag_family="tag36h11",
        tag_size_m=0.162,           # <-- make sure this matches your printed black square size
        min_decision_margin=25.0,
        fisheye_balance=0.3,
        outlier_trans_thresh_m=0.02,
        smooth_alpha=0.10,
        hold_seconds=0.7,
    )
    extr = ETGRSExtrinsicsEstimator(ext_cfg)

    # Network API subscriber
    sub = PupilSubscriber(
        topics=[
            "pupil.0.3d",
            "pupil.1.3d",
            "gaze",        # some versions publish "gaze"
            "gaze.",       # also catch "gaze.2d.0" etc
            "fixation",    # optional
        ]
    ).start()

    last_pupil0 = None
    last_pupil1 = None
    last_gaze_norm = None
    last_fix_norm = None

    if SHOW_WINDOWS:
        cv2.namedWindow("ETG (gaze overlay)", cv2.WINDOW_NORMAL)
        cv2.namedWindow("RS (raycast overlay)", cv2.WINDOW_NORMAL)

    print("\nRunning live 3D fixation...")
    print("If the ETG red dot doesn't match where you're looking, it's calibration/gaze-mapper, not raycasting.\n")
    print("Press q to quit.\n")

    last_print_t = 0.0

    with PupilStream(pupil_cfg) as pupil, RealSenseStream(rs_cfg) as rs_cam:
        while True:
            # --- grab latest network data quickly ---
            for _ in range(40):
                msg = sub.recv(timeout_ms=0)
                if msg is None:
                    break
                topic, payload = msg

                if topic == "pupil.0.3d":
                    last_pupil0 = payload
                elif topic == "pupil.1.3d":
                    last_pupil1 = payload
                elif topic.startswith("gaze"):
                    g = extract_gaze_norm_pos(topic, payload)
                    if g is not None:
                        last_gaze_norm = g
                elif topic.startswith("fixation"):
                    g = extract_gaze_norm_pos(topic, payload)
                    if g is not None:
                        last_fix_norm = g

            # choose what to use for ray:
            # prefer gaze; if missing, fallback to fixation norm_pos (coarser)
            gaze_norm = last_gaze_norm if last_gaze_norm is not None else last_fix_norm

            # --- read frames ---
            etg_bgr, _ = pupil.read()
            if etg_bgr is None:
                continue

            rs_color_bgr, rs_depth_m, rs_meta = rs_cam.read()
            if rs_color_bgr is None or rs_depth_m is None:
                continue

            K_rs = np.asarray(rs_meta["K_color"], dtype=np.float32).reshape(3, 3)
            dist_rs = np.asarray(rs_meta.get("dist_color", np.zeros((5, 1))), dtype=np.float32).reshape(-1, 1)

            # --- update extrinsics via tags ---
            ext_out = extr.update(
                etg_bgr=etg_bgr,
                rs_bgr=rs_color_bgr,
                K_etg=K_etg,
                dist_etg=dist_etg,
                K_rs=K_rs,
                dist_rs=dist_rs,
                now=time.time(),
            )
            T_etg_in_rs = ext_out["T_etg_in_rs"] if ext_out.get("ok", False) else None

            # --- compute ray + depth hit ---
            hit_rs = None
            u_etg = v_etg = None

            if gaze_norm is not None and T_etg_in_rs is not None:
                # 1) norm_pos -> pixel (Y flip)
                u_etg, v_etg = pupil_norm_to_pixel(gaze_norm, ETG_W, ETG_H)

                # 2) pixel -> undistorted bearing in ETG coords
                d_etg = gaze_pixel_to_unit_ray_etg(u_etg, v_etg, K_etg, dist_etg)

                # 3) transform ray into RS coords
                origin_rs, dir_rs = transform_ray_etg_to_rs(T_etg_in_rs, d_etg)

                # 4) raycast depth
                hit_rs = raycast_depth(origin_rs, dir_rs, rs_depth_m, K_rs)

            # --- visualization ---
            if SHOW_WINDOWS:
                # ETG overlay
                etg_vis = etg_bgr.copy()
                if u_etg is not None and v_etg is not None:
                    cv2.circle(etg_vis, (int(round(u_etg)), int(round(v_etg))), 10, (0, 0, 255), 2)
                    cv2.putText(etg_vis, "gaze", (int(u_etg) + 10, int(v_etg) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if last_gaze_norm is None:
                    cv2.putText(etg_vis, "NO gaze topic (enable gaze mapping in Pupil Capture)",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("ETG (gaze overlay)", etg_vis)

                # RS overlay
                rs_vis = rs_color_bgr.copy()

                if T_etg_in_rs is None:
                    cv2.putText(rs_vis, "NO T_etg_in_rs (AprilTag extrinsics not available)",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    t = ext_out.get("t_m", None)
                    rpy = ext_out.get("rpy_deg", None)
                    common = ext_out.get("common_ids", [])
                    kept = ext_out.get("kept", 0)
                    cv2.putText(rs_vis, "EXTRINSICS: OK",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    if t is not None and rpy is not None:
                        cv2.putText(rs_vis, f"T_etg_in_rs t=[{t[0]:+.3f},{t[1]:+.3f},{t[2]:+.3f}]m  rpy=[{rpy[0]:+.1f},{rpy[1]:+.1f},{rpy[2]:+.1f}]deg",
                                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(rs_vis, f"tags common={common} kept={kept} reason={ext_out.get('reason','')}",
                                    (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

                if hit_rs is not None:
                    uu, vv = project_rs_to_pixel(hit_rs, K_rs)
                    cv2.circle(rs_vis, (uu, vv), 10, (0, 255, 0), 2)
                    cv2.putText(rs_vis, f"HIT xyz=[{hit_rs[0]:+.3f},{hit_rs[1]:+.3f},{hit_rs[2]:+.3f}] m",
                                (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    if gaze_norm is None:
                        cv2.putText(rs_vis, "NO gaze/fixation norm_pos received",
                                    (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    elif T_etg_in_rs is not None:
                        cv2.putText(rs_vis, "Raycast MISS (no depth intersection)",
                                    (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("RS (raycast overlay)", rs_vis)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            # --- cleaner periodic terminal print (not spam) ---
            now = time.time()
            if now - last_print_t > 0.5:
                last_print_t = now

                # show pupils briefly
                if last_pupil0 is not None and last_pupil1 is not None:
                    p0 = last_pupil0.get("norm_pos", None)
                    p1 = last_pupil1.get("norm_pos", None)
                    c0 = last_pupil0.get("confidence", None)
                    c1 = last_pupil1.get("confidence", None)
                    print(f"p0={p0} conf={c0} | p1={p1} conf={c1}")

                if gaze_norm is not None:
                    print(f"gaze_norm={gaze_norm}  (u,v)={None if u_etg is None else (round(u_etg,1), round(v_etg,1))}")

                if hit_rs is not None:
                    print(f"HIT xyz_rs={hit_rs.tolist()}")
                else:
                    print("HIT: None")

    sub.stop()
    if SHOW_WINDOWS:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()