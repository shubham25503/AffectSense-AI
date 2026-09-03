"""
Blink Dynamics & Cognitive Restlessness Detector
================================================
Monitors Eye Aspect Ratio (EAR) over time to track blink rates,
flutter bursts (anxiety marker), and prolonged eyelid closures.
"""

from collections import deque
from dataclasses import dataclass
import time
from typing import Optional


@dataclass
class BlinkMetrics:
    blink_count: int
    blinks_per_minute: float
    is_currently_blinking: bool
    last_blink_duration_ms: float
    is_fluttering: bool     # Rapid successive blinks (nervousness / anxiety)
    is_prolonged: bool      # Prolonged eye closure (distress / fatigue)
    stress_indicator: float # 0.0 (calm) to 1.0 (high blink flutter/stress)


class BlinkDetector:
    """Tracks blinks and temporal ocular dynamics."""

    def __init__(
        self,
        ear_threshold: float = 0.19,
        consec_frames: int = 2,
        history_seconds: float = 30.0
    ):
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self.history_seconds = history_seconds

        self.blink_timestamps: deque = deque()
        self.blink_durations: deque = deque(maxlen=20)
        self.total_blinks = 0

        self.is_blinking = False
        self.blink_start_time = 0.0
        self.last_duration_ms = 0.0
        self.counter = 0

    def update(self, current_ear: float, timestamp: Optional[float] = None) -> BlinkMetrics:
        now = timestamp if timestamp is not None else time.time()

        # Remove blinks older than history window
        while self.blink_timestamps and (now - self.blink_timestamps[0] > self.history_seconds):
            self.blink_timestamps.popleft()

        is_prolonged = False
        if current_ear < self.ear_threshold:
            self.counter += 1
            if not self.is_blinking and self.counter >= self.consec_frames:
                self.is_blinking = True
                self.blink_start_time = now
            elif self.is_blinking and (now - self.blink_start_time > 0.45):
                is_prolonged = True
        else:
            if self.is_blinking:
                self.is_blinking = False
                self.total_blinks += 1
                self.blink_timestamps.append(now)
                self.last_duration_ms = max(50.0, (now - self.blink_start_time) * 1000.0)
                self.blink_durations.append(self.last_duration_ms)
            self.counter = 0

        # Calculate Blinks Per Minute (BPM)
        num_recent = len(self.blink_timestamps)
        # Scale to 60s
        effective_window = max(5.0, min(self.history_seconds, (now - self.blink_timestamps[0])) if self.blink_timestamps else 5.0)
        bpm = float((num_recent / effective_window) * 60.0)

        # Flutter detection: more than 3 blinks within 2.5 seconds
        recent_blinks_2s = sum(1 for t in self.blink_timestamps if (now - t) <= 2.5)
        is_fluttering = recent_blinks_2s >= 3 or bpm > 35.0

        # Stress indicator: normal resting is 12-20 BPM. Above 28 BPM is elevated.
        if bpm <= 18.0:
            stress_indicator = float(bpm / 36.0)
        else:
            stress_indicator = float(min(1.0, 0.5 + (bpm - 18.0) / 30.0))

        if is_fluttering:
            stress_indicator = max(stress_indicator, 0.8)

        return BlinkMetrics(
            blink_count=self.total_blinks,
            blinks_per_minute=round(bpm, 1),
            is_currently_blinking=self.is_blinking,
            last_blink_duration_ms=round(self.last_duration_ms, 1),
            is_fluttering=is_fluttering,
            is_prolonged=is_prolonged,
            stress_indicator=round(stress_indicator, 2),
        )

    def reset(self):
        self.blink_timestamps.clear()
        self.blink_durations.clear()
        self.total_blinks = 0
        self.is_blinking = False
        self.counter = 0
