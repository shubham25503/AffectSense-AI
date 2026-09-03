"""
Micro-Expression & FACS Action Unit Analyzer
============================================
Quantifies Facial Action Units (AU1, AU4, AU6, AU7, AU12, AU15, AU24, etc.),
smile symmetry, mouth aperture, and eye-mouth Duchenne synchronization.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ActionUnitScores:
    au1_inner_brow_raiser: float   # Grief / sadness marker
    au4_brow_lowerer: float        # Corrugator / frustration / concentration
    au6_cheek_raiser: float        # Duchenne genuine joy marker (crow's feet)
    au7_lid_tightener: float       # Eye squint / narrowing
    au12_lip_corner_puller: float  # Zygomaticus smile intensity
    au14_dimpler: float            # Smirk / unilateral pull
    au15_lip_corner_depressor: float # Sadness / frown
    au24_lip_pressor: float        # Emotional suppression / tension
    au5_upper_lid_raiser: float    # Fear / surprise / wide eyes
    smile_asymmetry: float         # 0.0 (symmetric) to 1.0 (unilateral smirk/contempt)
    duchenne_coherence: float      # Ratio of eye involvement to mouth smile
    mouth_aspect_ratio: float      # MAR: mouth openness


class MicroExpressionAnalyzer:
    """Extracts Facial Action Units (FACS) from MediaPipe blendshapes and 3D landmarks."""

    # Landmark indices for geometric verification
    LIP_LEFT = 61
    LIP_RIGHT = 291
    LIP_TOP = 0
    LIP_BOTTOM = 17

    BROW_INNER_LEFT = 107
    BROW_OUTER_LEFT = 70
    BROW_INNER_RIGHT = 336
    BROW_OUTER_RIGHT = 300

    CHEEK_LEFT = 117
    CHEEK_RIGHT = 346

    EYE_INNER_LEFT = 133
    EYE_OUTER_LEFT = 33
    EYE_INNER_RIGHT = 362
    EYE_OUTER_RIGHT = 263

    def process(
        self,
        landmarks_3d: np.ndarray,
        blendshapes_dict: Dict[str, float],
        avg_ear: float
    ) -> ActionUnitScores:
        """Analyze landmarks and blendshapes to compute calibrated Action Units."""

        # 1. AU12: Lip Corner Puller (Smile)
        bs_smile_l = blendshapes_dict.get("mouthSmileLeft", 0.0)
        bs_smile_r = blendshapes_dict.get("mouthSmileRight", 0.0)
        au12 = float((bs_smile_l + bs_smile_r) / 2.0)

        # 2. AU6: Cheek Raiser (Duchenne marker)
        bs_cheek_l = blendshapes_dict.get("cheekSquintLeft", 0.0)
        bs_cheek_r = blendshapes_dict.get("cheekSquintRight", 0.0)
        au6_raw = float((bs_cheek_l + bs_cheek_r) / 2.0)

        # 3. AU7: Lid Tightener (Orbicularis Oculi squint)
        bs_squint_l = blendshapes_dict.get("eyeSquintLeft", 0.0)
        bs_squint_r = blendshapes_dict.get("eyeSquintRight", 0.0)
        au7 = float((bs_squint_l + bs_squint_r) / 2.0)

        # Geometric cheek & eyelid narrowing factor:
        # In genuine smiles, eyes narrow (EAR drops significantly below baseline ~0.28)
        ear_squint_factor = float(np.clip((0.27 - avg_ear) / 0.10, 0.0, 1.0)) if avg_ear < 0.27 else 0.0
        au6 = float(max(au6_raw, au7 * 0.70, ear_squint_factor * 0.85))

        # 4. AU1: Inner Brow Raiser (Grief / Sadness)
        bs_brow_inner = blendshapes_dict.get("browInnerUp", 0.0)
        brow_in_l = landmarks_3d[self.BROW_INNER_LEFT]
        brow_out_l = landmarks_3d[self.BROW_OUTER_LEFT]
        brow_in_r = landmarks_3d[self.BROW_INNER_RIGHT]
        brow_out_r = landmarks_3d[self.BROW_OUTER_RIGHT]

        # Interocular normalization distance
        eye_l = landmarks_3d[self.EYE_INNER_LEFT]
        eye_r = landmarks_3d[self.EYE_INNER_RIGHT]
        interocular = max(1e-6, float(np.linalg.norm(eye_l[:2] - eye_r[:2])))

        # In grief/sadness, inner brow is pulled UP relative to outer brow
        tilt_l = (brow_out_l[1] - brow_in_l[1]) / interocular
        tilt_r = (brow_out_r[1] - brow_in_r[1]) / interocular
        avg_tilt = float((tilt_l + tilt_r) / 2.0)
        geom_au1 = float(np.clip((avg_tilt - 0.12) / 0.15, 0.0, 1.0))
        au1 = float(max(bs_brow_inner, geom_au1 * 0.75))

        # 5. AU4: Brow Lowerer (Corrugator / Furrow)
        bs_brow_down_l = blendshapes_dict.get("browDownLeft", 0.0)
        bs_brow_down_r = blendshapes_dict.get("browDownRight", 0.0)
        au4 = float((bs_brow_down_l + bs_brow_down_r) / 2.0)

        # 6. AU15: Lip Corner Depressor (Frown)
        bs_frown_l = blendshapes_dict.get("mouthFrownLeft", 0.0)
        bs_frown_r = blendshapes_dict.get("mouthFrownRight", 0.0)
        au15 = float((bs_frown_l + bs_frown_r) / 2.0)

        # 7. AU24: Lip Pressor (Suppression / tension)
        bs_press_l = blendshapes_dict.get("mouthPressLeft", 0.0)
        bs_press_r = blendshapes_dict.get("mouthPressRight", 0.0)
        au24 = float((bs_press_l + bs_press_r) / 2.0)

        # 8. AU5: Upper Lid Raiser (Wide eyes / surprise)
        bs_wide_l = blendshapes_dict.get("eyeWideLeft", 0.0)
        bs_wide_r = blendshapes_dict.get("eyeWideRight", 0.0)
        au5 = float((bs_wide_l + bs_wide_r) / 2.0)

        # 9. AU14 / Asymmetry (Dimpler / Smirk / Contempt)
        smile_diff = abs(bs_smile_l - bs_smile_r)
        au14 = float(np.clip(smile_diff * 1.5, 0.0, 1.0))
        smile_asymmetry = float(np.clip(smile_diff / max(1e-4, au12), 0.0, 1.0)) if au12 > 0.15 else 0.0

        # 10. MAR: Mouth Aspect Ratio
        lip_l = landmarks_3d[self.LIP_LEFT]
        lip_r = landmarks_3d[self.LIP_RIGHT]
        lip_top = landmarks_3d[self.LIP_TOP]
        lip_bot = landmarks_3d[self.LIP_BOTTOM]
        mar = float(np.linalg.norm(lip_top[:2] - lip_bot[:2]) / max(1e-6, np.linalg.norm(lip_l[:2] - lip_r[:2])))

        # 11. Duchenne Coherence:
        # High Duchenne: eyes squint organically and cheeks lift (AU6 > 0.22 or AU7 > 0.35).
        # Low Duchenne (Masked/Fake): mouth smiles wide (AU12 > 0.5) but cheeks remain flat (AU6 < 0.15) and eyes detached.
        if au12 > 0.15:
            base_engagement = float(max(au6, au7 * 0.85, ear_squint_factor))
            if avg_ear > 0.30 and au6 < 0.18:
                # Disconnected wide eyes with zero cheek elevation
                eye_engagement = float(max(0.0, base_engagement - (avg_ear - 0.30) * 2.5))
            else:
                eye_engagement = base_engagement

            duchenne_coherence = float(np.clip(eye_engagement / max(0.35, au12 * 0.85), 0.0, 1.0))
        else:
            duchenne_coherence = 1.0



        return ActionUnitScores(
            au1_inner_brow_raiser=round(au1, 3),
            au4_brow_lowerer=round(au4, 3),
            au6_cheek_raiser=round(au6, 3),
            au7_lid_tightener=round(au7, 3),
            au12_lip_corner_puller=round(au12, 3),
            au14_dimpler=round(au14, 3),
            au15_lip_corner_depressor=round(au15, 3),
            au24_lip_pressor=round(au24, 3),
            au5_upper_lid_raiser=round(au5, 3),
            smile_asymmetry=round(smile_asymmetry, 3),
            duchenne_coherence=round(duchenne_coherence, 3),
            mouth_aspect_ratio=round(mar, 3),
        )
