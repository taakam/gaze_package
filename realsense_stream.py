"""
realsense_stream.py

RealSense streaming helper (color + depth), with depth aligned to color.

Usage:
    from realsense_stream import RealSenseStream, RealSenseConfig

    with RealSenseStream(RealSenseConfig()) as rs_cam:
        while True:
            color_bgr, depth_m, meta = rs_cam.read()
            # color_bgr: (H,W,3) uint8 BGR
            # depth_m:  (H,W) float32 depth in meters, aligned to color
            # meta: dict with intrinsics, timestamp, etc.

Notes:
- Depth is aligned to the COLOR stream by default (recommended).
- Filters are optional; they can improve depth quality but add latency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pyrealsense2 as rs


@dataclass
class RealSenseConfig:
    # (W, H, FPS)
    color_wh_fps: Tuple[int, int, int] = (1280, 720, 30)
    depth_wh_fps: Tuple[int, int, int] = (848, 480, 30)

    # Stream options
    align_to_color: bool = True
    enable_filters: bool = True

    # Depth clamp / validity
    max_depth_m: float = 4.0

    # Device selection
    serial: Optional[str] = None  # set to a device serial string if you have multiple cameras


def K_from_intrinsics(intr: rs.intrinsics) -> np.ndarray:
    """Return 3x3 camera matrix from RealSense intrinsics."""
    K = np.array(
        [
            [intr.fx, 0.0, intr.ppx],
            [0.0, intr.fy, intr.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return K


def dist_from_intrinsics(intr: rs.intrinsics) -> np.ndarray:
    """Return distortion coefficients in OpenCV order [k1,k2,p1,p2,k3] if available."""
    # RealSense provides 5 coeffs for most models; if not, pad/truncate to 5.
    coeffs = np.array(intr.coeffs, dtype=np.float32).reshape(-1)
    if coeffs.size < 5:
        coeffs = np.pad(coeffs, (0, 5 - coeffs.size), mode="constant")
    elif coeffs.size > 5:
        coeffs = coeffs[:5]
    return coeffs


class RealSenseStream:
    def __init__(self, cfg: RealSenseConfig = RealSenseConfig()):
        self.cfg = cfg

        self.pipeline: Optional[rs.pipeline] = None
        self.profile: Optional[rs.pipeline_profile] = None
        self.align: Optional[rs.align] = None

        self.depth_scale: float = 0.001  # will be overwritten from sensor
        self._filters: Dict[str, Any] = {}

        # Cached intrinsics (of COLOR stream after start)
        self.color_intr: Optional[rs.intrinsics] = None
        self.depth_intr: Optional[rs.intrinsics] = None

    def __enter__(self) -> "RealSenseStream":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        if self.pipeline is not None:
            return

        self.pipeline = rs.pipeline()
        config = rs.config()

        if self.cfg.serial:
            config.enable_device(self.cfg.serial)

        cw, ch, cfps = self.cfg.color_wh_fps
        dw, dh, dfps = self.cfg.depth_wh_fps

        # COLOR + DEPTH streams
        config.enable_stream(rs.stream.color, cw, ch, rs.format.bgr8, cfps)
        config.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, dfps)

        self.profile = self.pipeline.start(config)

        # Depth scale (meters per depth unit)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())

        # Alignment
        self.align = rs.align(rs.stream.color) if self.cfg.align_to_color else None

        # Set up filters
        if self.cfg.enable_filters:
            # Typical useful filters; feel free to tweak later.
            self._filters = {
                "decimation": rs.decimation_filter(),
                "spatial": rs.spatial_filter(),
                "temporal": rs.temporal_filter(),
                "hole_filling": rs.hole_filling_filter(),
            }
            # Mild defaults (safe starting point)
            try:
                self._filters["decimation"].set_option(rs.option.filter_magnitude, 2)
            except Exception:
                pass
            try:
                self._filters["spatial"].set_option(rs.option.filter_magnitude, 2)
                self._filters["spatial"].set_option(rs.option.filter_smooth_alpha, 0.5)
                self._filters["spatial"].set_option(rs.option.filter_smooth_delta, 20)
            except Exception:
                pass
            try:
                self._filters["temporal"].set_option(rs.option.filter_smooth_alpha, 0.4)
                self._filters["temporal"].set_option(rs.option.filter_smooth_delta, 20)
            except Exception:
                pass

        # Cache intrinsics (from active streams)
        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_stream = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
        self.color_intr = color_stream.get_intrinsics()
        self.depth_intr = depth_stream.get_intrinsics()

        # Warm-up frames (auto-exposure etc.)
        for _ in range(10):
            self.pipeline.wait_for_frames()

    def stop(self) -> None:
        if self.pipeline is None:
            return
        try:
            self.pipeline.stop()
        finally:
            self.pipeline = None
            self.profile = None
            self.align = None

    def _apply_filters(self, depth_frame: rs.depth_frame) -> rs.depth_frame:
        if not self.cfg.enable_filters or not self._filters:
            return depth_frame
        f = depth_frame
        # Order matters
        for name in ("spatial", "temporal", "hole_filling"):
            filt = self._filters.get(name)
            if filt is not None:
                f = filt.process(f)
        return f

    def read(self, timeout_ms: int = 5000) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Returns:
            color_bgr: uint8 (H,W,3)
            depth_m:  float32 (H,W) depth in meters, aligned to color if cfg.align_to_color=True
            meta: dict with intrinsics + timestamp + depth_scale
        """
        if self.pipeline is None:
            raise RuntimeError("RealSenseStream not started. Use with RealSenseStream() as rs_cam: ... or call start().")

        frames = self.pipeline.wait_for_frames(timeout_ms)

        # Align depth -> color (so depth pixels match color pixels)
        if self.align is not None:
            frames = self.align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("Failed to get both depth and color frames from RealSense.")

        # Optionally filter depth (after alignment)
        depth_frame = self._apply_filters(depth_frame)

        # Convert to numpy
        color_bgr = np.asanyarray(color_frame.get_data())  # already BGR8 per config
        depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)  # uint16 -> float32

        # Convert to meters
        depth_m = depth_raw * self.depth_scale

        # Clamp invalid / too-far depth to 0
        if self.cfg.max_depth_m is not None and self.cfg.max_depth_m > 0:
            depth_m = np.where((depth_m > 0.0) & (depth_m <= self.cfg.max_depth_m), depth_m, 0.0).astype(np.float32)
        else:
            depth_m = np.where(depth_m > 0.0, depth_m, 0.0).astype(np.float32)

        # Intrinsics: if aligned_to_color, use COLOR intrinsics for projecting depth_m pixels
        # because depth_m pixels now correspond to color image pixels.
        # (depth_intr is still useful for debugging, but K_color is what you want for 2D->3D from aligned depth)
        color_intr = self.color_intr
        if color_intr is None:
            # fallback: query per-frame profile
            color_intr = color_frame.profile.as_video_stream_profile().get_intrinsics()

        K_color = K_from_intrinsics(color_intr)
        dist_color = dist_from_intrinsics(color_intr)

        meta: Dict[str, Any] = {
            "timestamp_ms": float(frames.get_timestamp()),
            "frame_number": int(color_frame.get_frame_number()),
            "depth_scale_m_per_unit": float(self.depth_scale),
            "color_intrinsics": {
                "width": int(color_intr.width),
                "height": int(color_intr.height),
                "fx": float(color_intr.fx),
                "fy": float(color_intr.fy),
                "ppx": float(color_intr.ppx),
                "ppy": float(color_intr.ppy),
                "model": str(color_intr.model),
                "coeffs": [float(x) for x in color_intr.coeffs],
            },
            "K_color": K_color,          # 3x3 float32
            "dist_color": dist_color,    # (5,) float32
            "aligned_depth_to_color": bool(self.cfg.align_to_color),
        }

        return color_bgr, depth_m, meta


if __name__ == "__main__":
    # Quick sanity test (press Ctrl+C to quit)
    import cv2

    cfg = RealSenseConfig()
    with RealSenseStream(cfg) as cam:
        while True:
            color, depth_m, meta = cam.read()
            # Visualize depth for debugging
            depth_vis = (np.clip(depth_m, 0, cfg.max_depth_m) / cfg.max_depth_m * 255.0).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            cv2.imshow("RealSense Color", color)
            cv2.imshow("RealSense Depth (aligned, meters)", depth_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    cv2.destroyAllWindows()
