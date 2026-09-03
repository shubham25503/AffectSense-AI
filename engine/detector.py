"""
SensoryPipeline: Master Human Emotion & True Sense Engine
=========================================================
Integrates MediaPipe 478-mesh landmark extraction, 3D retinal gaze tracking,
FACS action units, blink dynamics, rPPG pulse sensing, and sincerity classification.
"""

from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np

from .gaze_tracker import GazeRetinaTracker, GazeResult
from .micro_expressions import MicroExpressionAnalyzer, ActionUnitScores
from .blink_detector import BlinkDetector, BlinkMetrics
from .rppg_pulse import RPPGPulseSensor, PulseResult
from .sincerity_classifier import SincerityAffectClassifier, AffectDiagnosis


@dataclass
class SensoryResult:
    face_detected: bool
    gaze: GazeResult
    aus: ActionUnitScores
    blink: BlinkMetrics
    pulse: PulseResult
    affect: AffectDiagnosis
    landmarks_px: np.ndarray
    blendshapes_raw: Dict[str, float]
    frame_width: int
    frame_height: int


class SensoryPipeline:
    """End-to-end multi-modal affective and subconscious sense detector."""

    DEFAULT_MODEL_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "face_landmarker.task"
    )

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"MediaPipe Face Landmarker model not found at: {self.model_path}. "
                "Ensure models/face_landmarker.task exists."
            )

        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(
            model_asset_path=self.model_path,
            delegate=python.BaseOptions.Delegate.CPU
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)


        # Initialize sub-modules
        self.gaze_tracker = GazeRetinaTracker()
        self.micro_expressions = MicroExpressionAnalyzer()
        self.blink_detector = BlinkDetector()
        self.pulse_sensor = RPPGPulseSensor()
        self.classifier = SincerityAffectClassifier()

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp: Optional[float] = None
    ) -> Optional[SensoryResult]:
        """Process a single BGR video frame or static photo."""
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        detection_result = self.detector.detect(mp_image)

        if not detection_result.face_landmarks or len(detection_result.face_landmarks) == 0:
            return None

        # Extract 478 landmarks scaled to pixel dimensions
        raw_lms = detection_result.face_landmarks[0]
        pts = np.array([(p.x * w, p.y * h, p.z * w) for p in raw_lms], dtype=np.float32)

        # Extract raw blendshapes dict
        bs_dict: Dict[str, float] = {}
        if detection_result.face_blendshapes and len(detection_result.face_blendshapes) > 0:
            bs_dict = {
                cat.category_name: cat.score
                for cat in detection_result.face_blendshapes[0]
            }

        # 1. Gaze & Retina processing (fusing landmarks + blendshapes)
        gaze_res = self.gaze_tracker.process(pts, bs_dict)

        # 2. Blink dynamics
        blink_res = self.blink_detector.update(gaze_res.avg_ear, timestamp=timestamp)

        # 3. Micro-expressions & FACS
        aus_res = self.micro_expressions.process(pts, bs_dict, gaze_res.avg_ear)

        # 4. rPPG contactless pulse
        pulse_res = self.pulse_sensor.update(frame_bgr, pts, timestamp=timestamp)

        # 5. Sincerity and affective classification
        affect_res = self.classifier.evaluate(
            gaze=gaze_res,
            aus=aus_res,
            blink=blink_res,
            pulse=pulse_res
        )

        return SensoryResult(
            face_detected=True,
            gaze=gaze_res,
            aus=aus_res,
            blink=blink_res,
            pulse=pulse_res,
            affect=affect_res,
            landmarks_px=pts,
            blendshapes_raw=bs_dict,
            frame_width=w,
            frame_height=h,
        )

    def draw_hud(
        self,
        frame_bgr: np.ndarray,
        result: SensoryResult,
        show_mesh: bool = True,
        show_gaze_rays: bool = True
    ) -> np.ndarray:
        """Draws an augmented reality biometric HUD on the frame."""
        canvas = frame_bgr.copy()
        h, w, _ = canvas.shape
        pts = result.landmarks_px

        # 1. Retina / Iris 3D Gaze Vectors
        if show_gaze_rays:
            for is_left, eye in [(True, result.gaze.left_eye), (False, result.gaze.right_eye)]:
                ix, iy = int(eye.iris_center[0]), int(eye.iris_center[1])
                cv2.circle(canvas, (ix, iy), 5, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.circle(canvas, (ix, iy), 2, (100, 150, 240), -1, cv2.LINE_AA)

                ray_len = 40
                pitch_rad = np.radians(eye.gaze_pitch)
                yaw_rad = np.radians(eye.gaze_yaw)
                dx = int(ray_len * np.sin(yaw_rad))
                dy = int(-ray_len * np.sin(pitch_rad))
                target = (ix + dx, iy + dy)

                ray_color = (80, 100, 220) if eye.gaze_pitch < -5.0 else (100, 180, 100)
                cv2.arrowedLine(canvas, (ix, iy), target, ray_color, 2, tipLength=0.3)

        # 2. Key Facial Feature Contours (Subtle slate lines)
        if show_mesh:
            l_eye_idx = [33, 160, 158, 133, 153, 144, 33]
            r_eye_idx = [362, 385, 387, 263, 373, 380, 362]
            mouth_idx = [61, 81, 13, 311, 291, 402, 14, 178, 61]
            brow_l_idx = [70, 63, 105, 66, 107]
            brow_r_idx = [336, 296, 334, 293, 300]

            for idx_list, color in [
                (l_eye_idx, (180, 190, 200)),
                (r_eye_idx, (180, 190, 200)),
                (mouth_idx, (160, 170, 180)),
                (brow_l_idx, (140, 150, 170)),
                (brow_r_idx, (140, 150, 170)),
            ]:
                poly = np.array([pts[i][:2] for i in idx_list], dtype=np.int32)
                cv2.polylines(canvas, [poly], False, color, 1, cv2.LINE_AA)

        # 3. Simple Floating Context Badge Tracking Face
        min_x = int(np.min(pts[:, 0]))
        max_x = int(np.max(pts[:, 0]))
        min_y = int(np.min(pts[:, 1]))
        max_y = int(np.max(pts[:, 1]))
        cx = int((min_x + max_x) / 2)

        is_love = "Love" in result.affect.primary_state or "Affection" in result.affect.primary_state
        badge_color = (180, 100, 160) if is_love else ((60, 70, 200) if result.affect.is_masking_detected else ((70, 170, 90) if result.affect.sincerity_score >= 0.70 else (180, 150, 80)))
        tag_y = max(30, min_y - 20)
        tag_text = result.affect.primary_state
        (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_DUPLEX, 0.44, 1)

        box_x1 = max(10, cx - int(tw / 2) - 10)
        box_x2 = min(w - 10, box_x1 + tw + 20)
        box_y1 = tag_y - th - 10
        box_y2 = tag_y + 6

        # Translucent dark box
        fl_overlay = canvas.copy()
        cv2.rectangle(fl_overlay, (box_x1, box_y1), (box_x2, box_y2), (20, 28, 42), -1)
        cv2.addWeighted(fl_overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), (70, 85, 105), 1, cv2.LINE_AA)

        # Subtle connector line to forehead
        fh_pt = (int(pts[10][0]), int(pts[10][1]))
        cv2.line(canvas, fh_pt, (cx, box_y2), (90, 105, 125), 1, cv2.LINE_AA)
        cv2.putText(canvas, tag_text, (box_x1 + 10, tag_y), cv2.FONT_HERSHEY_DUPLEX, 0.44, (240, 245, 250), 1, cv2.LINE_AA)

        # 4. Clean Top Header
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, 75), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)
        cv2.line(canvas, (0, 75), (w, 75), (51, 65, 85), 1, cv2.LINE_AA)

        badge_title = "MASKED AFFECT" if result.affect.is_masking_detected else ("AUTHENTIC" if result.affect.sincerity_score >= 0.70 else "NEUTRAL")
        cv2.rectangle(canvas, (16, 12), (150, 32), badge_color, -1, cv2.LINE_AA)
        cv2.putText(canvas, badge_title, (22, 26), cv2.FONT_HERSHEY_DUPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        state_text = result.affect.primary_state
        cv2.putText(canvas, state_text, (16, 52), cv2.FONT_HERSHEY_DUPLEX, 0.52, (240, 245, 250), 1, cv2.LINE_AA)

        sub_text = f"Surface: {result.affect.surface_expression}  |  Truth: {result.affect.underlying_truth}"
        cv2.putText(canvas, sub_text, (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (148, 163, 184), 1, cv2.LINE_AA)

        # Sincerity Gauge (Top Right)
        s_score = result.affect.sincerity_score
        gauge_x = w - 210
        gauge_w = 190
        cv2.putText(canvas, f"Sincerity: {int(s_score * 100)}%", (gauge_x, 26), cv2.FONT_HERSHEY_DUPLEX, 0.44, (220, 230, 240), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (gauge_x, 36), (gauge_x + gauge_w, 46), (30, 41, 59), -1, cv2.LINE_AA)
        fill_w = int(gauge_w * s_score)
        fill_color = (70, 180, 90) if s_score >= 0.65 else ((60, 70, 220) if s_score < 0.40 else (70, 150, 220))
        cv2.rectangle(canvas, (gauge_x, 36), (gauge_x + fill_w, 46), fill_color, -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (gauge_x, 36), (gauge_x + gauge_w, 46), (71, 85, 105), 1, cv2.LINE_AA)



        # 4. Right Telemetry Sidebar
        sidebar_x = w - 260
        sidebar_y_start = 110
        sb_overlay = canvas.copy()
        cv2.rectangle(sb_overlay, (sidebar_x - 10, sidebar_y_start - 5), (w - 10, h - 20), (12, 15, 22), -1)
        cv2.addWeighted(sb_overlay, 0.70, canvas, 0.30, 0, canvas)
        cv2.rectangle(canvas, (sidebar_x - 10, sidebar_y_start - 5), (w - 10, h - 20), (60, 70, 85), 1, cv2.LINE_AA)

        cv2.putText(canvas, "OCULAR & FACS SENSORS", (sidebar_x, sidebar_y_start + 18), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 230, 255), 1, cv2.LINE_AA)

        metrics = [
            ("Retina Pitch (V)", f"{result.gaze.avg_pitch:+.1f}°", (0, 140, 255) if result.gaze.is_downward else (200, 220, 240)),
            ("Retina Yaw (H)", f"{result.gaze.avg_yaw:+.1f}°", (200, 220, 240)),
            ("Gaze Vector", result.gaze.gaze_direction[:18], (0, 255, 200)),
            ("Eye Aperture (EAR)", f"{result.gaze.avg_ear:.2f}", (200, 220, 240)),
            ("AU12 (Smile)", f"{result.aus.au12_lip_corner_puller:.2f}", (0, 220, 255)),
            ("AU6 (Duchenne Cheek)", f"{result.aus.au6_cheek_raiser:.2f}", (80, 220, 80) if result.aus.au6_cheek_raiser > 0.2 else (120, 120, 140)),
            ("AU7 (Eye Squint)", f"{result.aus.au7_lid_tightener:.2f}", (200, 220, 240)),
            ("AU1 (Grief Brow)", f"{result.aus.au1_inner_brow_raiser:.2f}", (0, 100, 255) if result.aus.au1_inner_brow_raiser > 0.18 else (120, 120, 140)),
            ("Duchenne Ratio", f"{result.aus.duchenne_coherence:.2f}", (80, 220, 80) if result.aus.duchenne_coherence > 0.55 else (0, 140, 255)),
            ("Blink Rate", f"{result.blink.blinks_per_minute:.0f} BPM", (0, 140, 255) if result.blink.is_fluttering else (200, 220, 240)),
            ("rPPG Pulse", f"{result.pulse.bpm:.0f} BPM", (255, 100, 120)),
        ]

        y_offset = sidebar_y_start + 45
        for label, val_str, color in metrics:
            cv2.putText(canvas, label, (sidebar_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 175, 195), 1, cv2.LINE_AA)
            cv2.putText(canvas, val_str, (sidebar_x + 160, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.40, color, 1, cv2.LINE_AA)
            y_offset += 24

        # Pulse Mini Graph at bottom of sidebar
        cv2.putText(canvas, f"PULSE WAVE ({result.pulse.status})", (sidebar_x, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 120, 140), 1, cv2.LINE_AA)
        wave = result.pulse.waveform[-25:]
        if len(wave) > 2:
            graph_y = y_offset + 35
            pts_wave = []
            for i, val in enumerate(wave):
                gx = sidebar_x + int(i * (230 / len(wave)))
                gy = int(graph_y - val * 12)
                pts_wave.append((gx, gy))
            cv2.polylines(canvas, [np.array(pts_wave, dtype=np.int32)], False, (0, 100, 255), 2, cv2.LINE_AA)

        return canvas

    def close(self):
        if hasattr(self, 'detector') and self.detector:
            self.detector.close()
