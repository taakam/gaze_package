#!/usr/bin/env python3
from __future__ import annotations

import time
from typing import Optional, Tuple, Dict, Any

import numpy as np
import cv2
import zmq
import msgpack

from realsense_stream import RealSenseStream, RealSenseConfig
from pupil_stream import PupilStream, PupilConfig
from etg_rs_extrinsics import ETGRSExtrinsicsEstimator, ExtrinsicsConfig
from board_pose import AprilTagBoardPoseEstimator, BoardConfig, inv_T
from target_sender import TargetSender

# -----------------------------
# USER CONFIG
# -----------------------------
PUPIL_REMOTE_IP = "127.0.0.1"
PUPIL_REMOTE_PORT = 50020

ETG_W = 640
ETG_H = 480

MIN_DEPTH_M = 0.15
MAX_DEPTH_M = 4.0

RAY_STEPS = 180
RAY_STEP_M = 0.015
HIT_THRESH_M = 0.025
NEIGHBORHOOD = 1

SHOW_WINDOWS = True

BOARD_TAG_SIZE_M = 0.162
BOARD_HORIZONTAL_INNER_GAP_M = 0.306
BOARD_VERTICAL_INNER_GAP_M = 0.024
BOARD_TAG_IDS = (0, 1, 2, 3)  # TL, TR, BL, BR

# robot spawn and board pose from sim
ROBOT_WORLD_XYZ = np.array([-0.95, -1.55, 1.015], dtype=np.float32)
BOARD_WORLD_XYZ = np.array([-0.211901, -1.55089, 1.0175], dtype=np.float32)

# board origin expressed in robot base frame
BOARD_IN_ROBOT = BOARD_WORLD_XYZ - ROBOT_WORLD_XYZ
CAPTURE_HOVER_Z_M = 0.08


def board_to_robot(hit_board: np.ndarray) -> np.ndarray:
    return BOARD_IN_ROBOT + hit_board


# -----------------------------
# Pupil Network API helpers
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
            topic_b, payload = self.sub.recv_multipart(
                flags=zmq.NOBLOCK if timeout_ms == 0 else 0
            )
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


def pupil_norm_to_pixel(norm_pos: Tuple[float, float], W: int, H: int) -> Tuple[float, float]:
    x, y = float(norm_pos[0]), float(norm_pos[1])
    x = 0.0 if x < 0 else (1.0 if x > 1 else x)
    y = 0.0 if y < 0 else (1.0 if y > 1 else y)
    u = x * (W - 1)
    v = (1.0 - y) * (H - 1)
    return u, v


def gaze_pixel_to_unit_ray_etg(u: float, v: float, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    d = np.asarray(dist, dtype=np.float64).reshape(-1)

    pts = np.array([[[u, v]]], dtype=np.float64)

    if d.size == 4:
        D = d.reshape(4, 1)
        und = cv2.fisheye.undistortPoints(pts, K, D)
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


def extract_gaze_norm_pos(topic: str, payload: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    if "norm_pos" in payload:
        try:
            a = payload["norm_pos"]
            return (float(a[0]), float(a[1]))
        except Exception:
            return None
    return None


def main():
    sender = TargetSender("tcp://127.0.0.1:5557")

    K_etg, dist_etg = load_world_intrinsics("world.intrinsics")

    rs_cfg = RealSenseConfig(
        color_wh_fps=(1280, 720, 30),
        depth_wh_fps=(848, 480, 30),
        align_to_color=True,
        enable_filters=False,
        max_depth_m=MAX_DEPTH_M,
    )

    pupil_cfg = PupilConfig(topic="frame.world", format="bgr", buffer_size=1)

    ext_cfg = ExtrinsicsConfig(
        tag_family="tag36h11",
        tag_size_m=BOARD_TAG_SIZE_M,
        min_decision_margin=25.0,
        fisheye_balance=0.3,
        outlier_trans_thresh_m=0.02,
        smooth_alpha=0.10,
        hold_seconds=0.7,
    )
    extr = ETGRSExtrinsicsEstimator(ext_cfg)

    board_cfg = BoardConfig(
        tag_family="tag36h11",
        tag_size_m=BOARD_TAG_SIZE_M,
        horizontal_inner_gap_m=BOARD_HORIZONTAL_INNER_GAP_M,
        vertical_inner_gap_m=BOARD_VERTICAL_INNER_GAP_M,
        tag_ids=BOARD_TAG_IDS,
        min_decision_margin=25.0,
    )
    board_est = AprilTagBoardPoseEstimator(board_cfg)

    sub = PupilSubscriber(
        topics=[
            "pupil.0.3d",
            "pupil.1.3d",
            "gaze",
            "gaze.",
            "fixation",
        ]
    ).start()

    last_gaze_norm = None
    last_fix_norm = None
    last_print_t = 0.0

    if SHOW_WINDOWS:
        cv2.namedWindow("ETG (gaze overlay)", cv2.WINDOW_NORMAL)
        cv2.namedWindow("RS (raycast overlay)", cv2.WINDOW_NORMAL)

    print("\nRunning live 3D fixation...")
    print("Press 'k' to send the current target to Docker bridge")
    print("Press 'q' or ESC to quit\n")

    try:
        with PupilStream(pupil_cfg) as pupil, RealSenseStream(rs_cfg) as rs_cam:
            while True:
                for _ in range(40):
                    msg = sub.recv(timeout_ms=0)
                    if msg is None:
                        break
                    topic, payload = msg

                    if topic.startswith("gaze"):
                        g = extract_gaze_norm_pos(topic, payload)
                        if g is not None:
                            last_gaze_norm = g
                    elif topic.startswith("fixation"):
                        g = extract_gaze_norm_pos(topic, payload)
                        if g is not None:
                            last_fix_norm = g

                gaze_norm = last_gaze_norm if last_gaze_norm is not None else last_fix_norm

                etg_bgr, _ = pupil.read()
                if etg_bgr is None:
                    continue

                rs_color_bgr, rs_depth_m, rs_meta = rs_cam.read()
                if rs_color_bgr is None or rs_depth_m is None:
                    continue

                K_rs = np.asarray(rs_meta["K_color"], dtype=np.float32).reshape(3, 3)
                dist_rs = np.asarray(
                    rs_meta.get("dist_color", np.zeros((5, 1))),
                    dtype=np.float32
                ).reshape(-1, 1)

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

                hit_rs = None
                u_etg = v_etg = None

                if gaze_norm is not None and T_etg_in_rs is not None:
                    u_etg, v_etg = pupil_norm_to_pixel(gaze_norm, ETG_W, ETG_H)
                    d_etg = gaze_pixel_to_unit_ray_etg(u_etg, v_etg, K_etg, dist_etg)
                    origin_rs, dir_rs = transform_ray_etg_to_rs(T_etg_in_rs, d_etg)
                    hit_rs = raycast_depth(origin_rs, dir_rs, rs_depth_m, K_rs)

                board_out = board_est.estimate(rs_color_bgr, K_rs, dist_rs)

                hit_board = None
                hit_robot = None
                target_robot = None

                if board_out.get("ok", False) and hit_rs is not None:
                    T_board_in_rs = board_out["T_board_in_cam"]
                    T_rs_in_board = inv_T(T_board_in_rs)

                    p_rs_h = np.array([hit_rs[0], hit_rs[1], hit_rs[2], 1.0], dtype=np.float32)
                    p_board_h = T_rs_in_board @ p_rs_h
                    hit_board = p_board_h[:3]

                    hit_robot = board_to_robot(hit_board)
                    target_robot = hit_robot.copy()
                    target_robot[2] += CAPTURE_HOVER_Z_M

                    target_robot[0] = float(np.clip(target_robot[0], 0.35, 0.95))
                    target_robot[1] = float(np.clip(target_robot[1], -0.35, 0.35))
                    target_robot[2] = float(np.clip(target_robot[2], 0.10, 0.50))

                if SHOW_WINDOWS:
                    etg_vis = etg_bgr.copy()
                    if u_etg is not None and v_etg is not None:
                        cv2.circle(etg_vis, (int(round(u_etg)), int(round(v_etg))), 10, (0, 0, 255), 2)
                        cv2.putText(
                            etg_vis, "gaze",
                            (int(u_etg) + 10, int(v_etg) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                        )
                    cv2.imshow("ETG (gaze overlay)", etg_vis)

                    rs_vis = rs_color_bgr.copy()

                    if hit_rs is not None:
                        uu, vv = project_rs_to_pixel(hit_rs, K_rs)
                        cv2.circle(rs_vis, (uu, vv), 10, (0, 255, 0), 2)
                        cv2.putText(
                            rs_vis,
                            f"HIT_rs=[{hit_rs[0]:+.3f},{hit_rs[1]:+.3f},{hit_rs[2]:+.3f}]",
                            (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
                        )

                    if hit_board is not None:
                        cv2.putText(
                            rs_vis,
                            f"HIT_board=[{hit_board[0]:+.3f},{hit_board[1]:+.3f},{hit_board[2]:+.3f}]",
                            (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2
                        )

                    if hit_robot is not None:
                        cv2.putText(
                            rs_vis,
                            f"HIT_robot=[{hit_robot[0]:+.3f},{hit_robot[1]:+.3f},{hit_robot[2]:+.3f}]",
                            (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2
                        )

                    if target_robot is not None:
                        cv2.putText(
                            rs_vis,
                            f"TARGET=[{target_robot[0]:+.3f},{target_robot[1]:+.3f},{target_robot[2]:+.3f}]",
                            (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
                        )

                    cv2.putText(
                        rs_vis,
                        "Press 'k' to send target",
                        (20, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2
                    )

                    cv2.imshow("RS (raycast overlay)", rs_vis)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                    elif key == ord("k"):
                        if target_robot is not None:
                            sender.send_target(
                                x=target_robot[0],
                                y=target_robot[1],
                                z=target_robot[2],
                                frame_id="base_link",
                            )
                            print("KEY 'k' PRESSED -> target sent to Docker bridge")
                        else:
                            print("KEY 'k' PRESSED but no valid target_robot available")

                now = time.time()
                if now - last_print_t > 0.5:
                    last_print_t = now

                    if hit_board is not None:
                        print(f"HIT_board  xyz=[{hit_board[0]:+.3f},{hit_board[1]:+.3f},{hit_board[2]:+.3f}]")
                    else:
                        print("HIT_board: None")

                    if hit_robot is not None:
                        print(f"HIT_robot  xyz=[{hit_robot[0]:+.3f},{hit_robot[1]:+.3f},{hit_robot[2]:+.3f}]")
                        print(f"TARGET     xyz=[{target_robot[0]:+.3f},{target_robot[1]:+.3f},{target_robot[2]:+.3f}]")
                    else:
                        print("HIT_robot: None")

    finally:
        sub.stop()
        sender.close()
        if SHOW_WINDOWS:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()