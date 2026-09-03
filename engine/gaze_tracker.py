"""
Gaze & Retina Tracker
=====================
Computes 3D iris center positions, retinal gaze vectors (pitch and yaw),
Eye Aspect Ratio (EAR), fixation stability, and saccadic restlessness.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class EyeMetrics:
    ear: float
    iris_center: Tuple[float, float, float]
    iris_x_ratio: float  # 0 (nasal/temporal) to 1.0
    iris_y_ratio: float  # 0 (top) to 1.0 (bottom)
    gaze_pitch: float    # degrees (- is downward, + is upward)
    gaze_yaw: float      # degrees (- is left, + is right)


@dataclass
class GazeResult:
    left_eye: EyeMetrics
    right_eye: EyeMetrics
    avg_ear: float
    avg_pitch: float     # degrees: < -6° indicates downward dejected gaze
    avg_yaw: float       # degrees: horizontal deflection
    gaze_direction: str  # "CENTER / ENGAGED", "DOWNWARD (DEJECTION/SHAME)", "LOOKING LEFT", "LOOKING RIGHT", "LOOKING UP"
    is_downward: bool
    is_direct_contact: bool
    is_averted: bool
    restlessness_index: float  # 0.0 (still/focused) to 1.0 (erratic/anxious saccades)
    fixation_score: float      # 0.0 to 1.0 (1.0 = highly steady lock)


class GazeRetinaTracker:
    """Tracks retinal position and gaze metrics from MediaPipe 478 landmarks."""

    # Left eye landmark indices
    LEFT_OUTER = 33
    LEFT_INNER = 133
    LEFT_TOP = 159
    LEFT_BOTTOM = 145
    LEFT_IRIS_CENTER = 468

    # Right eye landmark indices
    RIGHT_INNER = 362
    RIGHT_OUTER = 263
    RIGHT_TOP = 386
    RIGHT_BOTTOM = 374
    RIGHT_IRIS_CENTER = 473

    def __init__(self, history_len: int = 30):
        self.history_len = history_len
        self.gaze_history: deque = deque(maxlen=history_len)

    def _compute_single_eye(
        self,
        pts: np.ndarray,
        inner_idx: int,
        outer_idx: int,
        top_idx: int,
        bot_idx: int,
        iris_idx: int,
        is_left: bool
    ) -> EyeMetrics:
        inner = pts[inner_idx]
        outer = pts[outer_idx]
        top = pts[top_idx]
        bot = pts[bot_idx]
        iris = pts[iris_idx]

        # Horizontal and vertical dimensions
        eye_width = max(1e-6, float(np.linalg.norm(outer[:2] - inner[:2])))
        eye_height = max(1e-6, float(np.linalg.norm(top[:2] - bot[:2])))
        ear = float(eye_height / eye_width)

        # Iris coordinates relative to outer/inner corners
        min_x = min(inner[0], outer[0])
        max_x = max(inner[0], outer[0])
        x_span = max(1e-6, max_x - min_x)
        iris_x_ratio = float(np.clip((iris[0] - min_x) / x_span, 0.0, 1.0))

        # Vertical ratio: 0.0 at top eyelid, 1.0 at bottom eyelid
        min_y = min(top[1], bot[1])
        max_y = max(top[1], bot[1])
        y_span = max(1e-6, max_y - min_y)
        iris_y_ratio = float(np.clip((iris[1] - min_y) / y_span, 0.0, 1.0))

        # Approximate retinal gaze angles
        # Center resting ratio is roughly 0.50.
        # Ratio > 0.50 means iris is closer to bottom lid -> looking DOWN (negative pitch)
        pitch = float((0.48 - iris_y_ratio) * 65.0)

        # Yaw: For left eye, inner is right (higher x), outer is left (lower x).
        # For right eye, inner is left (lower x), outer is right (higher x).
        if is_left:
            yaw = float((iris_x_ratio - 0.52) * 60.0)
        else:
            yaw = float((iris_x_ratio - 0.48) * 60.0)

        return EyeMetrics(
            ear=ear,
            iris_center=(float(iris[0]), float(iris[1]), float(iris[2])),
            iris_x_ratio=iris_x_ratio,
            iris_y_ratio=iris_y_ratio,
            gaze_pitch=pitch,
            gaze_yaw=yaw,
        )

    def process(
        self,
        landmarks_3d: np.ndarray,
        blendshapes_dict: Optional[Dict[str, float]] = None
    ) -> GazeResult:
        """Process 478 3D landmarks (N, 3) and blendshapes."""
        left_eye = self._compute_single_eye(
            landmarks_3d,
            self.LEFT_INNER,
            self.LEFT_OUTER,
            self.LEFT_TOP,
            self.LEFT_BOTTOM,
            self.LEFT_IRIS_CENTER,
            is_left=True,
        )

        right_eye = self._compute_single_eye(
            landmarks_3d,
            self.RIGHT_INNER,
            self.RIGHT_OUTER,
            self.RIGHT_TOP,
            self.RIGHT_BOTTOM,
            self.RIGHT_IRIS_CENTER,
            is_left=False,
        )

        avg_ear = float((left_eye.ear + right_eye.ear) / 2.0)
        geom_pitch = float((left_eye.gaze_pitch + right_eye.gaze_pitch) / 2.0)
        geom_yaw = float((left_eye.gaze_yaw + right_eye.gaze_yaw) / 2.0)

        # Fuse with neural blendshapes if available
        if blendshapes_dict:
            down_l = blendshapes_dict.get("eyeLookDownLeft", 0.0)
            down_r = blendshapes_dict.get("eyeLookDownRight", 0.0)
            up_l = blendshapes_dict.get("eyeLookUpLeft", 0.0)
            up_r = blendshapes_dict.get("eyeLookUpRight", 0.0)
            in_l = blendshapes_dict.get("eyeLookInLeft", 0.0)
            in_r = blendshapes_dict.get("eyeLookInRight", 0.0)
            out_l = blendshapes_dict.get("eyeLookOutLeft", 0.0)
            out_r = blendshapes_dict.get("eyeLookOutRight", 0.0)

            # Neural pitch: positive is up, negative is down
            neural_pitch = float(((up_l + up_r) / 2.0 - (down_l + down_r) / 2.0) * 50.0)
            # Neural yaw: positive is right, negative is left
            neural_yaw = float(((in_l + out_r) / 2.0 - (in_r + out_l) / 2.0) * 45.0)

            avg_pitch = float(geom_pitch * 0.4 + neural_pitch * 0.6)
            avg_yaw = float(geom_yaw * 0.4 + neural_yaw * 0.6)
        else:
            avg_pitch = geom_pitch
            avg_yaw = geom_yaw

        # Track history for micro-saccadic restlessness
        self.gaze_history.append((avg_pitch, avg_yaw))

        if len(self.gaze_history) >= 5:
            arr = np.array(self.gaze_history)
            pitch_std = np.std(arr[:, 0])
            yaw_std = np.std(arr[:, 1])
            total_jitter = float(np.sqrt(pitch_std**2 + yaw_std**2))
            restlessness = float(np.clip(total_jitter / 10.0, 0.0, 1.0))
        else:
            restlessness = 0.1

        fixation_score = float(1.0 - restlessness)

        # Gaze direction classification
        # Note: In deep laughter squints (low EAR), lower lid occlusion can elevate eyeLookDown.
        # Downward gaze is true dejection when eyes are open or pitch is truly deflected downward.
        down_score = (blendshapes_dict.get("eyeLookDownLeft", 0.0) + blendshapes_dict.get("eyeLookDownRight", 0.0)) / 2.0 if blendshapes_dict else 0.0
        is_downward = avg_pitch < -6.0 or (down_score > 0.12 and avg_ear > 0.24) or avg_pitch < -4.5 and avg_ear > 0.28
        is_upward = avg_pitch > 7.0
        is_left = avg_yaw < -8.0
        is_right = avg_yaw > 8.0
        is_direct_contact = abs(avg_pitch) <= 5.5 and abs(avg_yaw) <= 7.5
        is_averted = is_downward or abs(avg_yaw) > 10.0

        if is_downward:
            gaze_dir = "DOWNWARD (DEJECTION/SHAME)"
        elif is_upward:
            gaze_dir = "LOOKING UP"
        elif is_left:
            gaze_dir = "LOOKING LEFT (AVERTED)"
        elif is_right:
            gaze_dir = "LOOKING RIGHT (AVERTED)"
        else:
            gaze_dir = "CENTER / ENGAGED"



        return GazeResult(
            left_eye=left_eye,
            right_eye=right_eye,
            avg_ear=avg_ear,
            avg_pitch=avg_pitch,
            avg_yaw=avg_yaw,
            gaze_direction=gaze_dir,
            is_downward=is_downward,
            is_direct_contact=is_direct_contact,
            is_averted=is_averted,
            restlessness_index=restlessness,
            fixation_score=fixation_score,
        )

    def reset(self):
        self.gaze_history.clear()
