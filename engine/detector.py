"""
SensoryPipeline: Master Human Emotion & True Sense Engine
=========================================================
Integrates MediaPipe 478-mesh landmark extraction, 3D retinal gaze tracking,
FACS action units, blink dynamics, rPPG pulse sensing, multi-face tracking,
and sincerity classification.
"""

from dataclasses import dataclass
import os
import time
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
    track_id: int = 1
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # min_x, min_y, max_x, max_y


class FaceTrack:
    """Maintains temporal identity and isolated biometric sensors for a specific person."""

    def __init__(self, track_id: int, bbox: Tuple[int, int, int, int], centroid: Tuple[float, float]):
        self.track_id = track_id
        self.bbox = bbox  # (min_x, min_y, max_x, max_y)
        self.centroid = centroid  # (cx, cy)
        self.last_seen_time = time.time()
        self.frames_unseen = 0

        # Isolated biometric state machines per person:
        self.gaze_tracker = GazeRetinaTracker()
        self.micro_expressions = MicroExpressionAnalyzer()
        self.blink_detector = BlinkDetector()
        self.pulse_sensor = RPPGPulseSensor()


class FaceTracker:
    """
    Multi-object face tracker assigning persistent IDs ('Person 1', 'Person 2', ...)
    across video frames using IoU and normalized centroid distance matching.
    """

    def __init__(self, max_unseen_frames: int = 20, max_centroid_dist_ratio: float = 0.35):
        self.tracks: Dict[int, FaceTrack] = {}
        self.next_track_id = 1
        self.max_unseen_frames = max_unseen_frames
        self.max_centroid_dist_ratio = max_centroid_dist_ratio

    def reset(self):
        """Reset all active tracks and reset ID counter to 1."""
        self.tracks.clear()
        self.next_track_id = 1

    @staticmethod
    def _compute_iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0
        area1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
        area2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
        union_area = area1 + area2 - inter_area
        return inter_area / max(1, union_area)

    def update(
        self,
        detected_items: List[Dict],
        frame_width: int,
        frame_height: int,
        is_static: bool = False
    ) -> List[Tuple[FaceTrack, Dict]]:
        """
        Matches detected faces to existing tracks or spawns new tracks.
        Returns a list of tuples: (FaceTrack, det_dict).
        """
        if not detected_items:
            for trk in self.tracks.values():
                trk.frames_unseen += 1
            self._purge_stale_tracks()
            return []

        # For static photos, order faces horizontally from left to right (Person 1, Person 2, ...)
        if is_static:
            sorted_detections = sorted(detected_items, key=lambda d: d['bbox'][0])
            matched_pairs = []
            for idx, det in enumerate(sorted_detections):
                tid = idx + 1
                track = FaceTrack(track_id=tid, bbox=det['bbox'], centroid=det['centroid'])
                matched_pairs.append((track, det))
            return matched_pairs

        # For video streams: track matching across consecutive frames
        for trk in self.tracks.values():
            trk.frames_unseen += 1

        active_track_ids = list(self.tracks.keys())
        diag = np.hypot(frame_width, frame_height)

        matched_track_ids = set()
        matched_det_indices = set()
        matched_pairs = []

        if active_track_ids:
            # Build cost matrix
            candidates = []
            for d_idx, det in enumerate(detected_items):
                d_cx, d_cy = det['centroid']
                d_box = det['bbox']
                for tid in active_track_ids:
                    trk = self.tracks[tid]
                    dist = np.hypot(trk.centroid[0] - d_cx, trk.centroid[1] - d_cy) / max(1.0, diag)
                    iou = self._compute_iou(trk.bbox, d_box)

                    # Similarity score calculation
                    if iou > 0.10:
                        sim = 0.6 * iou + 0.4 * max(0.0, 1.0 - (dist / self.max_centroid_dist_ratio))
                    elif dist < self.max_centroid_dist_ratio:
                        sim = max(0.0, 1.0 - (dist / self.max_centroid_dist_ratio))
                    else:
                        sim = 0.0

                    if sim > 0.20:
                        candidates.append((sim, tid, d_idx))

            # Sort candidates by similarity descending (greedy match)
            candidates.sort(key=lambda x: x[0], reverse=True)

            for sim, tid, d_idx in candidates:
                if tid in matched_track_ids or d_idx in matched_det_indices:
                    continue
                trk = self.tracks[tid]
                det = detected_items[d_idx]
                trk.bbox = det['bbox']
                trk.centroid = det['centroid']
                trk.frames_unseen = 0
                trk.last_seen_time = time.time()
                matched_track_ids.add(tid)
                matched_det_indices.add(d_idx)
                matched_pairs.append((trk, det))

        # Assign new tracks for unmatched detections
        for d_idx, det in enumerate(detected_items):
            if d_idx not in matched_det_indices:
                tid = self.next_track_id
                self.next_track_id += 1
                new_track = FaceTrack(track_id=tid, bbox=det['bbox'], centroid=det['centroid'])
                self.tracks[tid] = new_track
                matched_pairs.append((new_track, det))

        self._purge_stale_tracks()

        # Sort results by track ID for consistent reporting
        matched_pairs.sort(key=lambda pair: pair[0].track_id)
        return matched_pairs

    def _purge_stale_tracks(self):
        stale_ids = [
            tid for tid, trk in self.tracks.items()
            if trk.frames_unseen > self.max_unseen_frames
        ]
        for tid in stale_ids:
            del self.tracks[tid]


class SensoryPipeline:
    """End-to-end multi-modal affective and subconscious sense detector with multi-face tracking."""

    DEFAULT_MODEL_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "face_landmarker.task"
    )
    MODEL_DOWNLOAD_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

    def _ensure_model_exists(self):
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            print(f"[AffectSense] Downloading MediaPipe Face Landmarker model to {self.model_path}...")
            import urllib.request
            try:
                urllib.request.urlretrieve(self.MODEL_DOWNLOAD_URL, self.model_path)
                print("[AffectSense] Model download complete.")
            except Exception as e:
                raise FileNotFoundError(
                    f"MediaPipe Face Landmarker model not found at: {self.model_path} and automated download failed: {e}. "
                    f"Please place face_landmarker.task in {os.path.dirname(self.model_path)} manually."
                )

    def __init__(self, model_path: Optional[str] = None, max_faces: int = 5):
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.max_faces = max_faces
        self._ensure_model_exists()

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
            num_faces=self.max_faces
        )
        try:
            self.detector = vision.FaceLandmarker.create_from_options(options)
        except OSError as oe:
            import sys
            py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
            raise RuntimeError(
                f"MediaPipe FaceLandmarker failed to load C shared library (_dlopen error on Python {py_ver}): {oe}. "
                "This typically happens on Streamlit Cloud if the app is configured with Python 3.13+ or 3.14 (MediaPipe requires Python 3.10 or 3.11) "
                "or if Linux packages (libgl1, libglib2.0-0, libgomp1) are missing in packages.txt."
            ) from oe

        # Multi-face tracker
        self.face_tracker = FaceTracker()

        # Classification engine (stateless affect mapping)
        self.classifier = SincerityAffectClassifier()

        # Single-face sensors (used for backward compatibility / fallback)
        self.gaze_tracker = GazeRetinaTracker()
        self.micro_expressions = MicroExpressionAnalyzer()
        self.blink_detector = BlinkDetector()
        self.pulse_sensor = RPPGPulseSensor()

        # License & Access Key Security Manager
        try:
            from engine.auth import AuthManager
            self.auth_manager = AuthManager()
        except Exception:
            self.auth_manager = None

    def reset_tracker(self):
        """Resets multi-face tracks between video sessions."""
        self.face_tracker.reset()

    def process_frame_multi(
        self,
        frame_bgr: np.ndarray,
        timestamp: Optional[float] = None,
        is_static: bool = False
    ) -> List[SensoryResult]:
        """
        Process a BGR video frame or photo and return sensory results for all detected faces.
        Each face receives persistent tracking and isolated biometric state.
        """
        # Security Gating: Enforce active unexpired session
        if hasattr(self, "auth_manager") and self.auth_manager and self.auth_manager.auth_enabled:
            is_valid, rem, msg = self.auth_manager.is_current_session_valid()
            if not is_valid:
                raise PermissionError(
                    f"AffectSense Security Lockout: {msg} "
                    "Access is restricted to authorized sessions with an active Access Key."
                )

        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        detection_result = self.detector.detect(mp_image)

        if not detection_result.face_landmarks or len(detection_result.face_landmarks) == 0:
            self.face_tracker.update([], w, h, is_static=is_static)
            return []

        # Extract all detected face landmarks and blendshapes
        detected_items = []
        num_detected = len(detection_result.face_landmarks)
        for i in range(num_detected):
            raw_lms = detection_result.face_landmarks[i]
            pts = np.array([(p.x * w, p.y * h, p.z * w) for p in raw_lms], dtype=np.float32)

            bs_dict: Dict[str, float] = {}
            if detection_result.face_blendshapes and i < len(detection_result.face_blendshapes):
                bs_dict = {
                    cat.category_name: cat.score
                    for cat in detection_result.face_blendshapes[i]
                }

            min_x = max(0, int(np.min(pts[:, 0])))
            max_x = min(w, int(np.max(pts[:, 0])))
            min_y = max(0, int(np.min(pts[:, 1])))
            max_y = min(h, int(np.max(pts[:, 1])))
            bbox = (min_x, min_y, max_x, max_y)
            centroid = (float((min_x + max_x) / 2), float((min_y + max_y) / 2))

            detected_items.append({
                'bbox': bbox,
                'centroid': centroid,
                'pts': pts,
                'bs_dict': bs_dict
            })

        # Update tracker to get persistent FaceTrack objects
        matched_pairs = self.face_tracker.update(detected_items, w, h, is_static=is_static)

        results: List[SensoryResult] = []
        for track, det in matched_pairs:
            pts = det['pts']
            bs_dict = det['bs_dict']

            # Run isolated biometrics on this person's track
            gaze_res = track.gaze_tracker.process(pts, bs_dict)
            blink_res = track.blink_detector.update(gaze_res.avg_ear, timestamp=timestamp)
            aus_res = track.micro_expressions.process(pts, bs_dict, gaze_res.avg_ear)
            pulse_res = track.pulse_sensor.update(frame_bgr, pts, timestamp=timestamp)

            affect_res = self.classifier.evaluate(
                gaze=gaze_res,
                aus=aus_res,
                blink=blink_res,
                pulse=pulse_res
            )

            res = SensoryResult(
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
                track_id=track.track_id,
                bbox=track.bbox
            )
            results.append(res)

        return results

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp: Optional[float] = None
    ) -> Optional[SensoryResult]:
        """
        Process a single frame and return the primary face result.
        100% backward compatible with existing test suite and single-face pipelines.
        """
        results = self.process_frame_multi(frame_bgr, timestamp=timestamp, is_static=(timestamp is None))
        if not results:
            return None
        return results[0]

    def draw_hud_multi(
        self,
        frame_bgr: np.ndarray,
        results: List[SensoryResult],
        show_mesh: bool = True,
        show_gaze_rays: bool = True
    ) -> np.ndarray:
        """Draws an augmented reality biometric HUD for multiple detected and tracked faces."""
        if not results:
            return frame_bgr.copy()

        # If only 1 face detected, draw the detailed single-face HUD with sidebar
        if len(results) == 1:
            return self._draw_single_hud(frame_bgr, results[0], show_mesh, show_gaze_rays)

        canvas = frame_bgr.copy()
        h, w, _ = canvas.shape

        # Draw overlays for each tracked person
        for res in results:
            pts = res.landmarks_px
            pid = res.track_id
            affect = res.affect
            min_x, min_y, max_x, max_y = res.bbox
            cx = int((min_x + max_x) / 2)
            fw = max_x - min_x

            # Color coding for state
            is_love = "Love" in affect.primary_state or "Affection" in affect.primary_state
            if is_love:
                badge_color = (180, 100, 219)
            elif affect.is_masking_detected:
                badge_color = (60, 70, 220)  # Reddish
            elif affect.sincerity_score >= 0.70:
                badge_color = (70, 170, 90)   # Greenish
            else:
                badge_color = (180, 150, 80)  # Blue/neutral

            # 1. 3D Retinal Gaze Vectors
            if show_gaze_rays:
                for eye in [res.gaze.left_eye, res.gaze.right_eye]:
                    ix, iy = int(eye.iris_center[0]), int(eye.iris_center[1])
                    cv2.circle(canvas, (ix, iy), 4, (220, 220, 220), 1, cv2.LINE_AA)
                    cv2.circle(canvas, (ix, iy), 2, (100, 150, 240), -1, cv2.LINE_AA)

                    ray_len = 35
                    pitch_rad = np.radians(eye.gaze_pitch)
                    yaw_rad = np.radians(eye.gaze_yaw)
                    dx = int(ray_len * np.sin(yaw_rad))
                    dy = int(-ray_len * np.sin(pitch_rad))
                    target = (ix + dx, iy + dy)

                    ray_color = (80, 100, 220) if eye.gaze_pitch < -5.0 else (100, 180, 100)
                    cv2.arrowedLine(canvas, (ix, iy), target, ray_color, 2, tipLength=0.3)

            # 2. Key Facial Feature Contours (Subtle lines)
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

            # 3. Clean Corner Brackets around Face
            bl = max(14, int(fw * 0.12))
            cv2.line(canvas, (min_x, min_y), (min_x + bl, min_y), badge_color, 2, cv2.LINE_AA)
            cv2.line(canvas, (min_x, min_y), (min_x, min_y + bl), badge_color, 2, cv2.LINE_AA)

            cv2.line(canvas, (max_x, min_y), (max_x - bl, min_y), badge_color, 2, cv2.LINE_AA)
            cv2.line(canvas, (max_x, min_y), (max_x, min_y + bl), badge_color, 2, cv2.LINE_AA)

            cv2.line(canvas, (min_x, max_y), (min_x + bl, max_y), badge_color, 2, cv2.LINE_AA)
            cv2.line(canvas, (min_x, max_y), (min_x, max_y - bl), badge_color, 2, cv2.LINE_AA)

            cv2.line(canvas, (max_x, max_y), (max_x - bl, max_y), badge_color, 2, cv2.LINE_AA)
            cv2.line(canvas, (max_x, max_y), (max_x, max_y - bl), badge_color, 2, cv2.LINE_AA)

            # 4. Floating Diagnosis Badge Above Forehead
            tag_y = max(34, min_y - 22)
            tag_text = f"Person {pid}: {affect.primary_state} [{int(affect.sincerity_score * 100)}%]"
            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_DUPLEX, 0.42, 1)

            box_x1 = max(6, cx - int(tw / 2) - 8)
            box_x2 = min(w - 6, box_x1 + tw + 16)
            box_y1 = tag_y - th - 8
            box_y2 = tag_y + 6

            fl_overlay = canvas.copy()
            cv2.rectangle(fl_overlay, (box_x1, box_y1), (box_x2, box_y2), (18, 25, 38), -1)
            cv2.addWeighted(fl_overlay, 0.88, canvas, 0.12, 0, canvas)
            cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), badge_color, 1, cv2.LINE_AA)

            # Forehead connector line
            fh_pt = (int(pts[10][0]), int(pts[10][1]))
            cv2.line(canvas, fh_pt, (cx, box_y2), (90, 105, 125), 1, cv2.LINE_AA)
            cv2.putText(canvas, tag_text, (box_x1 + 8, tag_y), cv2.FONT_HERSHEY_DUPLEX, 0.42, (240, 245, 250), 1, cv2.LINE_AA)

            # 5. Mini Telemetry Strip Below Chin
            info_y = min(h - 14, max_y + 20)
            info_text = f"P{pid} | SINCERITY: {int(affect.sincerity_score * 100)}% | EAR: {res.gaze.avg_ear:.2f} | SMILE: {res.aus.au12_lip_corner_puller:.2f}"
            (itw, ith), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
            ibox_x1 = max(6, cx - int(itw / 2) - 6)
            ibox_x2 = min(w - 6, ibox_x1 + itw + 12)

            fl_overlay2 = canvas.copy()
            cv2.rectangle(fl_overlay2, (ibox_x1, info_y - ith - 4), (ibox_x2, info_y + 4), (15, 20, 30), -1)
            cv2.addWeighted(fl_overlay2, 0.85, canvas, 0.15, 0, canvas)
            cv2.rectangle(canvas, (ibox_x1, info_y - ith - 4), (ibox_x2, info_y + 4), (51, 65, 85), 1, cv2.LINE_AA)
            cv2.putText(canvas, info_text, (ibox_x1 + 6, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (203, 213, 225), 1, cv2.LINE_AA)

        # Top Multi-Face Header Banner
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, 54), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.90, canvas, 0.10, 0, canvas)
        cv2.line(canvas, (0, 54), (w, 54), (51, 65, 85), 1, cv2.LINE_AA)

        title_str = f"AFFECTSENSE AI | {len(results)} INDIVIDUALS TRACKED"
        cv2.putText(canvas, title_str, (16, 22), cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 230, 255), 1, cv2.LINE_AA)

        # Draw summary chips for each person across top header
        chip_x = 16
        for res in results:
            chip_text = f"P{res.track_id}: {res.affect.primary_state[:16]} ({int(res.affect.sincerity_score*100)}%)"
            (ctw, _), _ = cv2.getTextSize(chip_text, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)
            if chip_x + ctw + 12 < w - 10:
                p_color = (60, 70, 220) if res.affect.is_masking_detected else ((70, 170, 90) if res.affect.sincerity_score >= 0.70 else (180, 150, 80))
                cv2.rectangle(canvas, (chip_x, 30), (chip_x + ctw + 8, 48), (30, 41, 59), -1)
                cv2.rectangle(canvas, (chip_x, 30), (chip_x + ctw + 8, 48), p_color, 1)
                cv2.putText(canvas, chip_text, (chip_x + 4, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (241, 245, 249), 1, cv2.LINE_AA)
                chip_x += ctw + 14

        return canvas

    def _draw_single_hud(
        self,
        frame_bgr: np.ndarray,
        result: SensoryResult,
        show_mesh: bool = True,
        show_gaze_rays: bool = True
    ) -> np.ndarray:
        """Draws the rich augmented reality biometric HUD for a single face."""
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

        # 2. Key Facial Feature Contours
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
        tag_text = f"Person {result.track_id}: {result.affect.primary_state}" if result.track_id > 1 else result.affect.primary_state
        (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_DUPLEX, 0.44, 1)

        box_x1 = max(10, cx - int(tw / 2) - 10)
        box_x2 = min(w - 10, box_x1 + tw + 20)
        box_y1 = tag_y - th - 10
        box_y2 = tag_y + 6

        fl_overlay = canvas.copy()
        cv2.rectangle(fl_overlay, (box_x1, box_y1), (box_x2, box_y2), (20, 28, 42), -1)
        cv2.addWeighted(fl_overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), (70, 85, 105), 1, cv2.LINE_AA)

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

        # 5. Right Telemetry Sidebar
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

    def draw_hud(
        self,
        frame_bgr: np.ndarray,
        result: SensoryResult,
        show_mesh: bool = True,
        show_gaze_rays: bool = True
    ) -> np.ndarray:
        """Backward-compatible single face HUD drawer."""
        return self._draw_single_hud(frame_bgr, result, show_mesh=show_mesh, show_gaze_rays=show_gaze_rays)

    def close(self):
        if hasattr(self, 'detector') and self.detector:
            self.detector.close()
