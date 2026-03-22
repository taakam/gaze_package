#!/usr/bin/env python3
from __future__ import annotations

import zmq


class TargetSender:
    def __init__(self, address: str = "tcp://127.0.0.1:5557"):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUSH)
        self.sock.connect(address)

    def send_target(self, x: float, y: float, z: float, frame_id: str = "base_link"):
        self.sock.send_json({
            "frame_id": frame_id,
            "x": float(x),
            "y": float(y),
            "z": float(z),
        })

    def close(self):
        try:
            self.sock.close(0)
        except Exception:
            pass