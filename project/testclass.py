"""
test_brisk_pose_live.py  (UPDATED for new brisk_pose.py)

Live debug:
- Reads ETG (Pupil world) via your PupilStream.read()
- Reads RealSense color + aligned depth via RealSenseStream.read()
- Runs BRISK (on RS resized to ETG size) -> 2D-3D -> solvePnPRansac (EPnP) -> refine
- Displays match visualization + prints stats

Run:
    python3 test_brisk_pose_live.py
"""

import time
import numpy as np
import cv2
import msgpack

from realsense_stream import RealSenseStream, RealSenseConfig
from brisk_pose import estimate_pose_brisk, BriskPnPConfig, draw_inlier_matches
from pupil_stream import PupilStream, PupilConfig


def load_world_intrinsics(path: str = "world.intrinsics"):
    """Loads Pupil world cam intrinsics from msgpack (handles your string key '(640, 480)')."""
    with open(path, "rb") as fh:
        data = msgpack.unpack(fh, raw=False)

    # pick first key that isn't 'version'
    res_key = next(k for k in data.keys() if k != "version")
    block = data[res_key]

    # Support both spellings
    if "dist_coefs" in block:
        dist = np.array(block["dist_coefs"], dtype=np.float32).reshape(-1, 1)
    elif "dist_coeffs" in block:
        dist = np.array(block["dist_coeffs"], dtype=np.float32).reshape(-1, 1)
    else:
        raise KeyError(f"No distortion field found. Keys: {list(block.keys())}")

    K = np.array(block["camera_matrix"], dtype=np.float32)
    if K.shape != (3, 3):
        K = K.reshape(3, 3)

    print(f"[ETG] Using intrinsics key: {res_key} ({type(res_key)})")
    print("[ETG] K:\n", K)
    print("[ETG] dist:\n", dist.reshape(-1))
    return K, dist


def main():
    # ---- Load ETG intrinsics (640x480) ----
    K_etg, dist_etg = load_world_intrinsics("world.intrinsics")

    # ---- Start streams ----
    rs_cfg = RealSenseConfig(
        color_wh_fps=(1280, 720, 30),
        depth_wh_fps=(848, 480, 30),
        align_to_color=True,
        enable_filters=False,  # IMPORTANT: keep depth full-res for alignment/3D
        max_depth_m=4.0,
    )

    pupil_cfg = PupilConfig(topic="frame.world", format="bgr", buffer_size=1)

    # ---- BRISK+PnP config (forgiving first run) ----
    cfg = BriskPnPConfig()

    cfg.pnp_method = cv2.SOLVEPNP_AP3P  # works better with very few matches (vs EPnP)
    cfg.brisk_thresh = 20
    cfg.ratio_test = 0.85
    cfg.use_cross_check = False
    cfg.ransac_reproj_err_px = 20.0
    cfg.ransac_iters = 5000
    cfg.min_inliers = 10
    cfg.max_matches = 1000
    cfg.use_clahe = True
    cfg.return_debug = True

    # ---- OpenCV windows ----
    cv2.namedWindow("ETG (Pupil world)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("RealSense Color", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Matches (RS resized -> ETG)", cv2.WINDOW_NORMAL)

    with PupilStream(pupil_cfg) as pupil, RealSenseStream(rs_cfg) as rs_cam:
        fps_t0 = time.time()
        frames = 0

        while True:
            # 1) ETG frame
            etg_bgr, etg_meta = pupil.read()
            if etg_bgr is None:
                continue

            # 2) RealSense aligned RGB+depth
            rs_color_bgr, rs_depth_m, rs_meta = rs_cam.read()

            # Quick shape sanity (uncomment for debugging)
            # print("ETG:", etg_bgr.shape, "RS color:", rs_color_bgr.shape, "RS depth:", rs_depth_m.shape)

            # Provide K_rs to pose estimator
            cfg.K_rs = rs_meta["K_color"]

            # 3) Pose estimation
            res = estimate_pose_brisk(
                etg_bgr=etg_bgr,
                rs_color_bgr=rs_color_bgr,
                rs_depth_m_aligned=rs_depth_m,
                K_etg=K_etg,
                dist_etg=dist_etg,
                cfg=cfg,
            )

            # 4) Display raw frames
            cv2.imshow("ETG (Pupil world)", etg_bgr)
            cv2.imshow("RealSense Color", rs_color_bgr)

            # 5) Show matches + metrics
            if res.get("ok", False):
                dbg = res["debug"]

                vis = draw_inlier_matches(
                    etg_bgr=etg_bgr,
                    rs_color_small_bgr=dbg["rs_color_small"],
                    kp_etg=dbg["kp_etg"],
                    kp_rs_small=dbg["kp_rs_small"],
                    matches_good=dbg["matches_good"],
                    kept_match_indices=dbg["kept_match_indices"],
                    inliers=res["inliers"],
                    max_draw=80,
                )

                rmse = res["reproj_rmse_px"]
                rmse_txt = "rmse=?.??px" if rmse is None else f"rmse={rmse:.2f}px"
                txt = (
                    f"inliers={res['num_inliers']}  corr={res['num_corr']}  "
                    f"good={res['num_good_matches']}  {rmse_txt}"
                )
                cv2.putText(vis, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.imshow("Matches (RS resized -> ETG)", vis)

                if frames % 15 == 0:
                    print("[OK]", txt)
            else:
                err = res.get("error", "unknown error")
                blank = np.zeros((540, 1100, 3), dtype=np.uint8)
                cv2.putText(blank, f"POSE FAIL: {err}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                # Print extra stats if present
                stats = []
                for k in ("num_kp_etg", "num_kp_rs_small", "num_good_matches", "num_corr", "num_inliers"):
                    if k in res:
                        stats.append(f"{k}={res[k]}")
                if stats:
                    cv2.putText(blank, " | ".join(stats), (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

                cv2.imshow("Matches (RS resized -> ETG)", blank)

                if frames % 15 == 0:
                    print("[FAIL]", err, " ".join(stats))

            # FPS tick
            frames += 1
            if time.time() - fps_t0 > 2.0:
                fps = frames / (time.time() - fps_t0)
                fps_t0 = time.time()
                frames = 0
                # print(f"FPS: {fps:.1f}")

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
