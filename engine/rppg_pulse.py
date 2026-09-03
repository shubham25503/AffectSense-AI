"""
Remote Photoplethysmography (rPPG) Pulse Sensor
===============================================
Extracts subtle facial skin chrominance fluctuations from forehead/cheek
regions of interest to estimate contactless heart rate (BPM) and pulse wave.
"""

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from scipy.signal import butter, filtfilt


@dataclass
class PulseResult:
    bpm: float
    confidence: float
    waveform: List[float]  # Recent normalized pulse signal
    is_elevated: bool      # Stress marker (> 95 BPM)
    status: str            # "LOCKED", "CALIBRATING", "LOW CONFIDENCE"


class RPPGPulseSensor:
    """Estimates pulse rate using green channel micro-perfusion analysis."""

    # Forehead landmarks: 10, 109, 67, 103, 54, 21, 162
    FOREHEAD_LANDMARKS = [10, 109, 67, 103, 284, 297, 338]

    def __init__(self, buffer_size: int = 120, fps: float = 30.0):
        self.buffer_size = buffer_size
        self.fps = fps
        self.raw_signals: deque = deque(maxlen=buffer_size)
        self.timestamps: deque = deque(maxlen=buffer_size)
        self.last_bpm = 72.0
        self.last_confidence = 0.0

    def _extract_forehead_roi_mean(
        self,
        frame_bgr: np.ndarray,
        landmarks_3d: np.ndarray
    ) -> Optional[float]:
        h, w, _ = frame_bgr.shape
        # Get forehead centroid
        fh_pts = landmarks_3d[self.FOREHEAD_LANDMARKS]
        cx = int(np.mean(fh_pts[:, 0]))
        cy = int(np.mean(fh_pts[:, 1]))

        # Bounding box of 30x30 around forehead center
        box_w = max(10, int(w * 0.04))
        box_h = max(10, int(h * 0.03))
        x1 = max(0, cx - box_w)
        x2 = min(w, cx + box_w)
        y1 = max(0, cy - box_h)
        y2 = min(h, cy + box_h)

        if x2 <= x1 or y2 <= y1:
            return None

        roi = frame_bgr[y1:y2, x1:x2]
        # Extract mean green channel (blood hemoglobin has highest absorption in green)
        mean_g = float(np.mean(roi[:, :, 1]))
        return mean_g

    def update(
        self,
        frame_bgr: np.ndarray,
        landmarks_3d: np.ndarray,
        timestamp: Optional[float] = None
    ) -> PulseResult:
        import time
        t = timestamp if timestamp is not None else time.time()
        val = self._extract_forehead_roi_mean(frame_bgr, landmarks_3d)

        if val is not None:
            self.raw_signals.append(val)
            self.timestamps.append(t)

        if len(self.raw_signals) < 45:
            return PulseResult(
                bpm=self.last_bpm,
                confidence=0.0,
                waveform=[0.0] * 30,
                is_elevated=False,
                status="CALIBRATING (NEED MORE FRAMES)",
            )

        # Dynamic FPS calculation if timestamps available
        if len(self.timestamps) >= 2:
            dt = (self.timestamps[-1] - self.timestamps[0]) / (len(self.timestamps) - 1)
            effective_fps = max(10.0, min(60.0, 1.0 / dt)) if dt > 0 else self.fps
        else:
            effective_fps = self.fps

        # Detrend and normalize
        sig = np.array(self.raw_signals)
        sig = sig - np.mean(sig)

        # Butterworth bandpass filter (0.75 Hz to 2.5 Hz -> 45 to 150 BPM)
        lowcut = 0.75
        highcut = 2.5
        nyq = 0.5 * effective_fps
        low = min(lowcut / nyq, 0.9)
        high = min(highcut / nyq, 0.95)

        try:
            if low < high and len(sig) > 20:
                b, a = butter(2, [low, high], btype='band')
                filtered = filtfilt(b, a, sig)
            else:
                filtered = sig
        except Exception:
            filtered = sig

        # Compute FFT power spectrum
        n = len(filtered)
        freqs = np.fft.rfftfreq(n, d=1.0 / effective_fps)
        fft_vals = np.abs(np.fft.rfft(filtered))

        # Restrict to human pulse range (45 - 150 BPM)
        valid_idx = np.where((freqs >= lowcut) & (freqs <= highcut))[0]
        if len(valid_idx) > 0:
            peak_idx = valid_idx[np.argmax(fft_vals[valid_idx])]
            peak_freq = freqs[peak_idx]
            raw_bpm = peak_freq * 60.0

            # Confidence is peak power ratio relative to band power
            band_power = np.sum(fft_vals[valid_idx])
            peak_power = fft_vals[peak_idx]
            conf = float(np.clip(peak_power / max(1e-6, band_power / 3.0), 0.0, 1.0))

            # Exponential smoothing
            self.last_bpm = round(float(self.last_bpm * 0.8 + raw_bpm * 0.2), 1)
            self.last_confidence = round(conf, 2)
            status = "LOCKED" if conf > 0.4 else "ACQUIRING"
        else:
            status = "LOW SIGNAL"

        norm_wave = (filtered[-30:] / max(1e-6, np.std(filtered))).tolist()

        return PulseResult(
            bpm=self.last_bpm,
            confidence=self.last_confidence,
            waveform=norm_wave,
            is_elevated=self.last_bpm > 95.0,
            status=status,
        )

    def reset(self):
        self.raw_signals.clear()
        self.timestamps.clear()
        self.last_bpm = 72.0
        self.last_confidence = 0.0
