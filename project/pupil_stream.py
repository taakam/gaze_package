"""
Pupil Core / Pupil Capture streaming helper for frame.world (and optionally eye cams).

Requires:
    pip install pupil-labs-pupil-core-network-client

Usage:
    from pupil_stream import PupilStream, PupilConfig

    with PupilStream(PupilConfig(topic="frame.world")) as ps:
        frame_bgr, meta = ps.read()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pupil_labs.pupil_core_network_client as pcnc


@dataclass
class PupilConfig:
    address: str = "127.0.0.1"
    port: int = 50020
    topic: str = "frame.world"      # "frame.world", "frame.eye.0", "frame.eye.1"
    format: str = "bgr"             # "bgr" is what you want for OpenCV
    buffer_size: int = 1            # keep latest
    require_format_match: bool = True


class PupilStream:
    def __init__(self, cfg: Optional[PupilConfig] = None):
        self.cfg = cfg or PupilConfig()
        self.device: Optional[pcnc.Device] = None
        self._sub_ctx = None
        self._sub = None

    # -------------------------
    # Lifecycle
    # -------------------------
    def start(self) -> "PupilStream":
        if self.device is not None:
            return self

        self.device = pcnc.Device(self.cfg.address, self.cfg.port)

        # Ask Pupil Capture/Service to publish frames in BGR (OpenCV-friendly)
        self.device.send_notification(
            {"subject": "frame_publishing.set_format", "format": self.cfg.format}
        )

        # Subscribe in background (buffer_size=1 means always newest frame)
        self._sub_ctx = self.device.subscribe_in_background(
            self.cfg.topic, buffer_size=self.cfg.buffer_size
        )
        self._sub = self._sub_ctx.__enter__()

        return self

    def stop(self) -> None:
        if self._sub_ctx is not None:
            try:
                self._sub_ctx.__exit__(None, None, None)
            except Exception:
                pass
        self._sub_ctx = None
        self._sub = None
        self.device = None

    def __enter__(self) -> "PupilStream":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # -------------------------
    # Decode
    # -------------------------
    @staticmethod
    def decode_frame(payload: Dict[str, Any]) -> np.ndarray:
        """
        Robust decode for pcnc message.payload.

        Common payload keys:
            topic, width, height, index, timestamp, format, __raw_data__
        __raw_data__[0] is usually:
            - np.ndarray (H,W,3) uint8   (common)
            - tuple-wrapped array
            - bytes/memoryview (less common)
        """
        if "__raw_data__" not in payload:
            raise KeyError(f"Missing __raw_data__. Keys={list(payload.keys())}")

        w = int(payload["width"])
        h = int(payload["height"])
        fmt = str(payload.get("format", "bgr")).lower()

        img = payload["__raw_data__"][0]

        # Sometimes wrapped
        if isinstance(img, tuple):
            img = img[0]

        # Already ndarray -> done
        if isinstance(img, np.ndarray):
            return img

        # bytes / memoryview fallback
        if isinstance(img, memoryview):
            img = img.tobytes()

        if isinstance(img, (bytes, bytearray)):
            if fmt == "bgr":
                return np.frombuffer(img, dtype=np.uint8).reshape(h, w, 3)
            elif fmt == "gray":
                return np.frombuffer(img, dtype=np.uint8).reshape(h, w)
            else:
                raise ValueError(f"Unsupported format for raw bytes: {fmt}")

        raise TypeError(f"Unsupported __raw_data__ type: {type(img)}")

    # -------------------------
    # Public read()
    # -------------------------
    def read(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Returns:
            frame: np.ndarray (BGR) for topic
            meta: dict with timestamp, index, width/height, topic, format
        """
        if self._sub is None:
            raise RuntimeError("PupilStream not started. Use .start() or a with-block.")

        msg = self._sub.recv_new_message()
        payload = msg.payload

        if self.cfg.require_format_match:
            fmt = str(payload.get("format", "")).lower()
            if fmt and fmt != self.cfg.format.lower():
                raise ValueError(f"Frame format mismatch: got {fmt}, expected {self.cfg.format}")

        frame = self.decode_frame(payload)
        meta = {
            "topic": payload.get("topic"),
            "timestamp": payload.get("timestamp"),
            "index": payload.get("index"),
            "format": payload.get("format"),
            "width": payload.get("width"),
            "height": payload.get("height"),
        }
        return frame, meta
