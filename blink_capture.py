#!/usr/bin/env python3
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class BlinkCaptureConfig:
    blink_gap_s: float = 0.65          # max time between blinks
    sequence_timeout_s: float = 2.0    # reset if too slow
    confidence_threshold: float = 0.6  # pupil confidence threshold
    required_blinks: int = 3


class TripleBlinkCapture:
    """
    Detects a triple-blink event from low-confidence pupil packets.

    Assumes a blink corresponds to pupil confidence dropping below threshold.
    Uses edge detection so one blink counts once.
    """

    def __init__(self, cfg: BlinkCaptureConfig):
        self.cfg = cfg
        self._eyes_closed = False
        self._blink_count = 0
        self._first_blink_t: Optional[float] = None
        self._last_blink_t: Optional[float] = None

    def reset(self):
        self._eyes_closed = False
        self._blink_count = 0
        self._first_blink_t = None
        self._last_blink_t = None

    def update(self, pupil0_conf: Optional[float], pupil1_conf: Optional[float], now: Optional[float] = None) -> bool:
        """
        Returns True exactly once when triple blink is detected.
        """
        if now is None:
            now = time.time()

        confs = [c for c in (pupil0_conf, pupil1_conf) if c is not None]
        if len(confs) == 0:
            return False

        # blink if both eyes low confidence, or single available eye low
        if len(confs) == 2:
            eyes_closed_now = (confs[0] < self.cfg.confidence_threshold and confs[1] < self.cfg.confidence_threshold)
        else:
            eyes_closed_now = (confs[0] < self.cfg.confidence_threshold)

        # timeout reset
        if self._first_blink_t is not None and (now - self._first_blink_t) > self.cfg.sequence_timeout_s:
            self.reset()

        detected = False

        # rising edge: open -> closed counts as one blink
        if eyes_closed_now and not self._eyes_closed:
            if self._last_blink_t is not None and (now - self._last_blink_t) > self.cfg.blink_gap_s:
                self.reset()

            if self._blink_count == 0:
                self._first_blink_t = now

            self._blink_count += 1
            self._last_blink_t = now

            if self._blink_count >= self.cfg.required_blinks:
                detected = True
                self.reset()

        self._eyes_closed = eyes_closed_now
        return detected