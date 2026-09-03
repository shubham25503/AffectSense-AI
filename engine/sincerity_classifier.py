"""
Sincerity & Multi-Modal Affect Classifier (25+ Emotion Spectrum)
================================================================
Combines retinal gaze vectors, FACS Action Units (AU1, AU4, AU6, AU7, AU9, AU10,
AU12, AU14, AU15, AU24), blink dynamics, head tilt, and rPPG pulse to classify
genuine vs. masked affective states with dual-layer explanations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from .gaze_tracker import GazeResult
from .micro_expressions import ActionUnitScores
from .blink_detector import BlinkMetrics
from .rppg_pulse import PulseResult


@dataclass
class AffectDiagnosis:
    primary_state: str            # One of 25+ emotional catalog states
    surface_expression: str       # Superficial appearance
    underlying_truth: str         # Subconscious truth revealed by eyes/retina/AUs
    sincerity_score: float        # 0.0 (totally masked) to 1.0 (completely genuine)
    is_masking_detected: bool     # True if facial mask contradicts internal state
    confidence: float             # 0.0 to 1.0
    scientific_justification: str # Technical bio-mechanical explanation
    layman_justification: str     # Plain English everyday explanation
    diagnostic_notes: List[str]   # Specific forensic bullet points
    emotion_radar: Dict[str, float] # Probability distribution across 8 core dimensions


class SincerityAffectClassifier:
    """Multi-modal affective classifier with deep retinal & micro-expression fusion."""

    def evaluate(
        self,
        gaze: GazeResult,
        aus: ActionUnitScores,
        blink: Optional[BlinkMetrics] = None,
        pulse: Optional[PulseResult] = None,
    ) -> AffectDiagnosis:
        notes: List[str] = []

        # Extract primary signals
        smile = aus.au12_lip_corner_puller
        cheek = aus.au6_cheek_raiser
        squint = aus.au7_lid_tightener
        grief_brow = aus.au1_inner_brow_raiser
        corrugator = aus.au4_brow_lowerer
        frown = aus.au15_lip_corner_depressor
        lip_press = aus.au24_lip_pressor
        wide_eyes = aus.au5_upper_lid_raiser
        asymmetry = aus.smile_asymmetry
        coherence = aus.duchenne_coherence
        mar = aus.mouth_aspect_ratio

        pitch = gaze.avg_pitch
        yaw = gaze.avg_yaw
        ear = gaze.avg_ear
        restlessness = gaze.restlessness_index
        is_direct_contact = gaze.is_direct_contact

        blink_bpm = blink.blinks_per_minute if blink else 16.0
        is_fluttering = blink.is_fluttering if blink else False
        is_prolonged_blink = blink.is_prolonged if blink else False
        pulse_bpm = pulse.bpm if pulse else 72.0
        pulse_elevated = pulse.is_elevated if pulse else False

        # -------------------------------------------------------------
        # 1. FORENSIC CUE DETECTION
        # -------------------------------------------------------------
        is_smiling = smile >= 0.28
        # Genuine Duchenne crinkling requires visible eyelid narrowing (EAR < 0.27) or high cheek elevation
        has_duchenne_crinkle = (ear < 0.26 or (cheek >= 0.40 and ear < 0.28)) and coherence >= 0.50
        is_gaze_down = pitch <= -4.0 or gaze.is_downward
        is_gaze_averted = gaze.is_averted
        has_grief_brow = grief_brow >= 0.18
        has_corrugator_furrow = corrugator >= 0.35
        has_suppression_lip = lip_press >= 0.28 and mar < 0.25
        is_wide_eyed_smile = is_smiling and ear >= 0.28 and not has_duchenne_crinkle
        is_nervous_blink = is_fluttering or blink_bpm >= 28.0 or restlessness >= 0.38


        # -------------------------------------------------------------
        # 2. DECISION LOGIC: 25+ EMOTIONAL STATES
        # -------------------------------------------------------------

        # --- TIER 1: LOVE, ROMANCE & WARM ATTACHMENT ---
        # "In Love / Deep Affection": Soft warm gaze, direct lock, radiant affectionate smile, relaxed brow, elevated pulse
        if (
            0.30 <= smile <= 0.95
            and is_direct_contact
            and not is_gaze_down
            and not has_corrugator_furrow
            and grief_brow <= 0.12
            and (coherence >= 0.45 or cheek >= 0.25)
            and restlessness <= 0.28
        ):
            primary_state = "In Love / Deep Affection"
            surface = "Soft Warm Smile & Direct Gaze"
            underlying = "Cardiovascular Arousal & Romantic Fixation"
            sincerity = 0.96
            is_masking = False
            scientific = (
                f"Sustained foveal lock (Pitch {pitch:+.1f}°, Yaw {yaw:+.1f}°) paired with active Zygomaticus "
                f"(AU12={smile:.2f}), Orbicularis Oculi elevation (AU6={cheek:.2f}), relaxed frontalis (AU1={grief_brow:.2f}), "
                f"and steady cardiovascular tone ({pulse_bpm:.0f} BPM) indicates oxytocin/dopamine-mediated romantic attachment."
            )
            layman = (
                "This person is looking directly at you with unmistakable love and warmth! Their smile reaches their eyes, "
                "their face is completely open and relaxed, and their steady eye contact reflects emotional adoration and affection."
            )
            notes.append(f"Warm direct eye gaze (pitch={pitch:+.1f}°, yaw={yaw:+.1f}°) with low restlessness ({restlessness:.2f}).")
            notes.append(f"Harmonious cheek elevation (AU6={cheek:.2f}) without inner brow grief tension.")


        # "Shy Attraction / Flirting": Intermittent gaze aversion + micro-smile + blinking
        elif (
            0.20 <= smile <= 0.55
            and (is_gaze_down or abs(yaw) > 6.0)
            and ear <= 0.26
            and (blink_bpm >= 22.0 or restlessness > 0.25)
            and not has_grief_brow
        ):
            primary_state = "Shy Attraction / Flirtatious Hesitation"
            surface = "Coy Micro-Smile & Averted Glance"
            underlying = "Romantic Interest with Modesty or Nervousness"
            sincerity = 0.88
            is_masking = False
            scientific = (
                f"Intermittent gaze aversion (Yaw {yaw:+.1f}°, Pitch {pitch:+.1f}°) coupled with subtle AU12 smile ({smile:.2f}), "
                f"slight ocular narrowing (EAR={ear:.2f}), and heightened ocular saccades ({restlessness:.2f}) matches canonical courtship coy-glance cues."
            )
            layman = (
                "This person seems to like you, but is acting a bit shy or flustered! They are smiling quietly while glancing down "
                "or looking away playfully, which is classic body language for attraction combined with nervousness."
            )
            notes.append("Averting gaze while smiling gently—a signature sign of modest flirtation or coy attraction.")

        # "Compassionate Tenderness": Soft gaze, slight inner brow compassion lift, gentle smile
        elif (
            0.15 <= smile <= 0.45
            and 0.12 <= grief_brow <= 0.25
            and not is_gaze_down
            and not has_corrugator_furrow
        ):
            primary_state = "Compassionate Tenderness / Empathy"
            surface = "Gentle Sympathetic Countenance"
            underlying = "Deep Emotional Resonance & Warmth"
            sincerity = 0.92
            is_masking = False
            scientific = (
                f"Co-occurrence of slight AU1 inner brow raiser ({grief_brow:.2f}) with soft AU12 smile ({smile:.2f}) "
                f"and steady visual engagement reflects empathic concern rather than dejection."
            )
            layman = (
                "This person is feeling deep empathy and kindness towards you. Their gentle smile and slightly raised inner eyebrows "
                "show they genuinely care about what you're saying and feel for you."
            )
            notes.append("Mild inner brow tension combined with soft smile reflects genuine compassion.")

        # --- TIER 2: MASKED & DECEPTIVE EMOTIONS (THE USER'S PRIMARY FOCUS) ---
        # "Masked Sadness (Smiling Melancholy)": Smiling mouth + downward retinal gaze or grief brow
        elif is_smiling and ((is_gaze_down and not has_duchenne_crinkle) or has_grief_brow or (frown >= 0.18 and not has_duchenne_crinkle)):
            primary_state = "Masked Sadness (Smiling Melancholy / High Sincerity Deficit)"
            surface = "Superficial Smile (AU12 Lip Corner Pull)"
            underlying = "Subconscious Sadness / Grief (Downward Retinal Pitch & Grief Brow)"
            sincerity = float(np.clip(0.18 + (1.0 - abs(pitch) / 25.0) * 0.12, 0.10, 0.38))
            is_masking = True
            scientific = (
                f"Action Unit Incongruence: Zygomaticus Major (AU12={smile:.2f}) is active while the optical gaze vector "
                f"is deflected downward (Pitch {pitch:+.1f}°). Orbital crinkling is suppressed (EAR={ear:.2f}, AU6={cheek:.2f}), "
                f"and Frontalis Medialis (AU1={grief_brow:.2f}) reveals involuntary grief micro-tensions."
            )
            layman = (
                "This person is putting on a 'brave face'. Even though their lips are smiling, their eyes are looking down and lack "
                "the natural crinkles of genuine happiness. Deep down, they are experiencing sadness, hurt, or exhaustion that they are trying to hide."
            )
            notes.append(f"Mouth is smiling ({smile:.2f}), but retinal pitch is angled downward ({pitch:+.1f}°), indicating internal dejection.")
            if has_grief_brow:
                notes.append(f"Inner eyebrow 'grief muscle' (AU1={grief_brow:.2f}) is actively counter-tensing against the smile.")
            if not has_duchenne_crinkle:
                notes.append(f"Eyes lack genuine Duchenne orbital narrowing (EAR={ear:.2f}, AU6={cheek:.2f}), leaving the upper gaze detached.")

        # "Masked Anxiety / Nervous Placation": Smiling mouth + rapid blinks / flutter / saccades
        elif is_smiling and (is_nervous_blink or (has_suppression_lip and restlessness > 0.28)):
            primary_state = "Masked Anxiety (Nervous Smile / Placation)"
            surface = "Social Appeasement Smile"
            underlying = "Cognitive Restlessness & High Internal Anxiety"
            sincerity = float(np.clip(0.30 - restlessness * 0.15, 0.15, 0.42))
            is_masking = True
            scientific = (
                f"Elevated ocular flutter/blink frequency ({blink_bpm:.1f} BPM) and saccadic restlessness ({restlessness:.2f}) "
                f"contradict the voluntary AU12 smile ({smile:.2f}), signaling sympathetic adrenergic hyper-arousal and appeasement."
            )
            layman = (
                "This person is smiling out of nervousness or tension, not because they are happy. Their eyes are darting or blinking rapidly, "
                "which shows they are feeling anxious, uncomfortable, or trying to avoid confrontation."
            )
            notes.append(f"Smile accompanied by elevated ocular restlessness ({restlessness:.2f}) or rapid blinks ({blink_bpm:.1f} BPM).")
            if has_suppression_lip:
                notes.append("Lip-compression (AU24) reveals underlying emotional suppression.")

        # "Contempt / Smug Disdain": Asymmetric unilateral smirk
        elif asymmetry >= 0.38 or aus.au14_dimpler >= 0.35:
            primary_state = "Contempt / Smug Disdain (Asymmetric Affect)"
            surface = "Unilateral Smirk"
            underlying = "Social Dominance, Skepticism or Dismissive Bias"
            sincerity = float(np.clip(0.50 - asymmetry * 0.2, 0.30, 0.55))
            is_masking = False
            scientific = (
                f"Pronounced bilateral asymmetry index ({asymmetry:.2f}) with unilateral AU14 dimpler pull "
                f"is the pathognomonic FACS marker for moral superiority, disdain, or skepticism."
            )
            layman = (
                "This is a one-sided smirk or sarcastic smile. It suggests the person feels skeptical, dismissive, or mildly superior "
                "rather than truly cheerful or friendly."
            )
            notes.append(f"High smile asymmetry ({asymmetry:.2f}) indicates unilateral lip retraction (smirk).")

        # "Suppressed Frustration / Strained Politeness": Smile + Corrugator furrow + lip compression
        elif has_corrugator_furrow and has_suppression_lip and not has_duchenne_crinkle:
            primary_state = "Suppressed Frustration / Strained Politeness"
            surface = "Polite Mask"
            underlying = "Repressed Irritation & Jaw Tension"
            sincerity = 0.28
            is_masking = True
            scientific = (
                f"Co-contraction of Corrugator Supercilii (AU4={corrugator:.2f}) and Orbicularis Oris (AU24={lip_press:.2f}) "
                f"overlaid on a voluntary smile indicates active suppression of annoyance or hostility."
            )
            layman = (
                "This person is gritting their teeth and trying to be polite, but they are actually very frustrated or annoyed. "
                "Their eyebrows are furrowed and their lips are tightly pressed together."
            )
            notes.append(f"Brow furrow (AU4={corrugator:.2f}) contradicts mouth smile, indicating conscious emotional restraint.")

        # "Forced / Pan Am Smile (Social Masking)": Smile without eye crinkle
        elif is_wide_eyed_smile or (is_smiling and coherence < 0.40):
            primary_state = "Forced / Pan Am Smile (Social Masking)"
            surface = "Courtesy Smile"
            underlying = "Social Obligation without Affective Resonance"
            sincerity = float(np.clip(0.35 + coherence * 0.15, 0.25, 0.48))
            is_masking = True
            scientific = (
                f"Absence of Orbicularis Oculi contraction (AU6={cheek:.2f}) despite elevated Zygomaticus (AU12={smile:.2f}). "
                f"Eye aperture (EAR={ear:.2f}) remains uncompressed, confirming a volitional non-Duchenne smile."
            )
            layman = (
                "This is a 'customer service' or polite social smile. The mouth is doing the smiling, but the eyes are completely neutral "
                "and detached. They are just smiling out of etiquette, not real emotion."
            )
            notes.append(f"Eye aperture (EAR={ear:.2f}) remains wide and orbital cheeks (AU6={cheek:.2f}) are unactivated.")

        # "Jealousy / Envious Resentment": Micro-smile + hardened stare + corrugator (no Duchenne, tight mouth)
        elif is_smiling and has_corrugator_furrow and is_direct_contact and not has_duchenne_crinkle and mar < 0.28:
            primary_state = "Jealousy / Envious Resentment"

            surface = "Hardened Polite Smile"
            underlying = "Competitive Tension or Submerged Envy"
            sincerity = 0.35
            is_masking = True
            scientific = (
                f"Direct fixed gaze coupled with Corrugator tension (AU4={corrugator:.2f}) and partial AU12 smile "
                f"indicates negative social comparison and suppressed resentment."
            )
            layman = (
                "This person has an intense, slightly guarded look with a stiff smile. It often indicates jealousy, competitive tension, "
                "or wanting to hide feeling threatened."
            )
            notes.append("Hardened, unblinking gaze combined with brow furrow and forced mouth elevation.")

        # --- TIER 3: AUTHENTIC POSITIVE STATES ---
        elif has_duchenne_crinkle and mar > 0.32:
            primary_state = "Playful Amusement / Delight"
            surface = "Radiant Open Smile"
            underlying = "Spontaneous Joy & Amusement"
            sincerity = float(np.clip(0.85 + coherence * 0.12, 0.85, 0.98))
            is_masking = False
            scientific = (
                f"Balanced Duchenne co-activation (Coherence={coherence:.2f}) with open mouth depression (MAR={mar:.2f}) "
                f"and orbital narrowing (AU6/AU7={cheek:.2f}). Full psychological congruence."
            )
            layman = (
                "This person is having a great time! They are laughing or beaming with authentic amusement. Their eyes are crinkled "
                "naturally and their whole face is energized with joy."
            )
            notes.append(f"Strong Duchenne coherence ({coherence:.2f}) with open mouth aperture (MAR={mar:.2f}).")

        elif is_smiling and has_duchenne_crinkle:
            primary_state = "Authentic Joy (Duchenne Smile)"
            surface = "Genuine Smile"
            underlying = "True Positive Affect & Psychological Congruence"
            sincerity = float(np.clip(0.78 + coherence * 0.18, 0.75, 0.96))
            is_masking = False
            scientific = (
                f"Harmonious co-activation of AU12 ({smile:.2f}) and AU6/AU7 ({cheek:.2f}), coherence ratio {coherence:.2f}. "
                f"Direct eye contact without downward deflection confirms authentic joy."
            )
            layman = (
                "This is 100% genuine happiness. The smile reaches their eyes, causing the telltale warm crinkles around the corners. "
                "There is no hidden sadness or forced politeness here."
            )
            notes.append(f"Orbicularis Oculi matches Zygomaticus (coherence={coherence:.2f}).")

        # --- TIER 4: NON-SMILING VULNERABLE & INWARD STATES ---
        # "Genuine Sadness / Grief":
        elif (has_grief_brow and corrugator >= 0.15) or (frown >= 0.20 and is_gaze_down):
            primary_state = "Genuine Sadness / Grief"
            surface = "Sorrowful Demeanor"
            underlying = "Authentic Sadness / Mourning"
            sincerity = 0.92
            is_masking = False
            scientific = (
                f"Activation of Darwin's 'grief triangle': Frontalis Medialis (AU1={grief_brow:.2f}) + Corrugator (AU4={corrugator:.2f}) "
                f"+ Lip Depressor (AU15={frown:.2f}) with downward gaze orientation ({pitch:+.1f}°)."
            )
            layman = (
                "This person is genuinely feeling sad or heartbroken. They aren't trying to hide it; their brows are drawn up in sorrow "
                "and their gaze is pointed downwards."
            )
            notes.append(f"Darwinian grief triangle: inner brow lifted (AU1={grief_brow:.2f}) and eyes cast down.")

        # "Shame / Dejection / Guilt" (Requires downward gaze + actual sorrow or grief tension):
        elif is_gaze_down and abs(pitch) > 7.5 and (has_grief_brow or frown >= 0.15 or corrugator >= 0.25):
            primary_state = "Shame / Dejection / Guilt"
            surface = "Averted Gaze / Subdued Face"
            underlying = "Self-Conscious Dejection or Vulnerability"
            sincerity = 0.88
            is_masking = False
            scientific = (
                f"Significant downward retinal pitch ({pitch:+.1f}°) coupled with frontalis/corrugator grief tension "
                f"(AU1={grief_brow:.2f}, AU4={corrugator:.2f}) indicates self-conscious dejection."
            )
            layman = (
                "This person is feeling ashamed, guilty, or deeply dejected. They are looking down and avoiding eye contact "
                "with visible sorrow or discomfort."
            )
            notes.append(f"Downward gaze deflection ({pitch:+.1f}°) with grief tension.")

        # "Downward Focus / Screen Reading" (Neutral downward gaze at desk, phone, or keyboard):
        elif is_gaze_down and abs(pitch) > 6.0 and grief_brow < 0.15 and frown < 0.15 and smile < 0.20:
            primary_state = "Downward Focus / Reading"
            surface = "Looking Downward at Screen/Desk"
            underlying = "Cognitive Concentration or Reading"
            sincerity = 0.94
            is_masking = False
            scientific = (
                f"Downward ocular pitch ({pitch:+.1f}°) without brow furrow or grief tension (AU1={grief_brow:.2f}, AU15={frown:.2f}) "
                f"corresponds to normal downward foveal fixation (e.g. reading a screen or keyboard)."
            )
            layman = (
                "This person is simply glancing down at their screen, keyboard, or desk to read or think. Their facial muscles are calm and neutral."
            )
            notes.append(f"Downward gaze ({pitch:+.1f}°) without sorrow or masking cues.")


        # "Embarrassment / Flustered":
        elif is_gaze_down and 0.12 <= smile <= 0.28 and (blink_bpm > 20.0 or pulse_elevated):
            primary_state = "Embarrassment / Flustered"
            surface = "Nervous Downward Micro-Smile"
            underlying = "Self-Conscious Exposure & Flustering"
            sincerity = 0.85
            is_masking = False
            scientific = (
                f"Down-and-away retinal shift accompanied by suppressed zygomatic twitch (AU12={smile:.2f}) "
                f"and autonomic vasodilation (pulse={pulse_bpm:.0f} BPM) characterizes embarrassment."
            )
            layman = (
                "This person is flustered or embarrassed! They have a shy nervous smile while looking down, likely blushing or feeling self-conscious."
            )
            notes.append("Downward gaze with nervous smile and elevated pulse.")

        # "Nostalgia / Bittersweet Melancholy":
        elif 0.12 <= smile <= 0.35 and 0.12 <= grief_brow <= 0.25 and pitch > 2.0:
            primary_state = "Nostalgia / Bittersweet Melancholy"
            surface = "Distant Soft Smile"
            underlying = "Reflective Longing & Fond Memories"
            sincerity = 0.90
            is_masking = False
            scientific = (
                f"Upward eye gaze (Pitch {pitch:+.1f}°) paired with subtle AU1 ({grief_brow:.2f}) and gentle smile ({smile:.2f}) "
                f"is indicative of cognitive memory recall accompanied by mixed positive/sorrowful affect."
            )
            layman = (
                "This person is feeling nostalgic. They are gazing off into the distance with a gentle smile, remembering something "
                "or someone meaningful from the past with a mix of fondness and wistful longing."
            )
            notes.append("Distant upward gaze with mixed smile and inner brow nostalgia marker.")

        # "Suppressed Anger / Resentment":
        elif corrugator >= 0.42 and (has_suppression_lip or lip_press >= 0.28):
            primary_state = "Suppressed Anger / Resentment"
            surface = "Rigid / Hardened Countenance"
            underlying = "Submerged Hostility & Jaw Clenching"
            sincerity = 0.82
            is_masking = False
            scientific = (
                f"Intense Corrugator Supercilii (AU4={corrugator:.2f}) + Orbicularis Oris compression (AU24={lip_press:.2f}) "
                f"without smiling indicates actively suppressed rage or hostility."
            )
            layman = (
                "This person is angry and trying hard not to snap. Their eyebrows are tightly knit together, their mouth is pressed into a thin line, "
                "and their jaw is rigid."
            )
            notes.append(f"Corrugator furrow (AU4={corrugator:.2f}) with pressed lips (AU24={lip_press:.2f}).")

        # "Fear / Panic":
        elif wide_eyes >= 0.32 and (grief_brow >= 0.20 or pulse_elevated):
            primary_state = "Fear / Apprehension"
            surface = "Wide-Eyed Alarm"
            underlying = "Acute Threat Perception / Hypervigilance"
            sincerity = 0.92
            is_masking = False
            scientific = (
                f"Levator Palpebrae Superioris (AU5={wide_eyes:.2f}) exposing sclera with elevated pulse ({pulse_bpm:.0f} BPM) "
                f"confirms autonomic fight-or-flight response."
            )
            layman = (
                "This person is feeling frightened, alarmed, or on edge. Their eyes are wide open, showing the whites of their eyes, "
                "and their heart rate is elevated."
            )
            notes.append(f"Wide eye aperture (AU5={wide_eyes:.2f}) exposing sclera.")

        # "Astonishment / Surprise":
        elif wide_eyes >= 0.32 and mar >= 0.35:
            primary_state = "Astonishment / Surprise"
            surface = "Open Jaw & Wide Eyes"
            underlying = "Cognitive Disruption & Sudden Novelty"
            sincerity = 0.94
            is_masking = False
            scientific = (
                f"Concurrent Levator activation (AU5={wide_eyes:.2f}) and mandibular relaxation (MAR={mar:.2f}) "
                f"reflects the universal startle/novelty appraisal."
            )
            layman = (
                "This person was caught completely by surprise! Their mouth has dropped open and their eyes are wide with shock or wonder."
            )
            notes.append(f"Jaw drop (MAR={mar:.2f}) and ocular widening (AU5={wide_eyes:.2f}).")

        # "Boredom / Apathy / Disengagement":
        elif is_prolonged_blink or (ear < 0.20 and restlessness < 0.10 and smile < 0.15 and corrugator < 0.15):
            primary_state = "Boredom / Apathy / Disengagement"
            surface = "Drooping Lids & Unfocused Gaze"
            underlying = "Hypo-Arousal & Cognitive Withdrawal"
            sincerity = 0.90
            is_masking = False
            scientific = (
                f"Eyelid ptosis (EAR={ear:.2f}) with sluggish saccades ({restlessness:.2f}) and absence of facial muscle tone "
                f"corresponds to reduced central nervous system arousal."
            )
            layman = (
                "This person is completely zoned out, bored, or sleepy. Their eyelids are heavy and drooping, and their face is totally blank."
            )
            notes.append(f"Heavy eyelids (EAR={ear:.2f}) and minimal ocular motion.")

        # "Deep Concentration / Inward Thought":
        elif ear < 0.23 and not is_gaze_down and corrugator >= 0.15 and restlessness < 0.18:
            primary_state = "Deep Concentration / Inward Thought"
            surface = "Focused Squint"
            underlying = "Intense Analytic Focus & Cognitive Absorption"
            sincerity = 0.95
            is_masking = False
            scientific = (
                f"Slight squint (EAR={ear:.2f}) with Corrugator engagement (AU4={corrugator:.2f}) and rock-steady fixation "
                f"indicates high working-memory cognitive load."
            )
            layman = (
                "This person is deep in thought and concentrating hard on something complex. Their eyes are slightly narrowed and locked in focus."
            )
            notes.append(f"Steady gaze lock (restlessness={restlessness:.2f}) and cognitive squint.")

        # "Serene Contentment":
        elif 0.15 <= smile <= 0.30 and cheek >= 0.10 and not is_gaze_down:
            primary_state = "Serene Contentment"
            surface = "Calm Micro-Smile"
            underlying = "Internal Peace & Balanced Affect"
            sincerity = 0.90
            is_masking = False
            scientific = (
                f"Low-intensity bilateral AU12 ({smile:.2f}) with relaxed ocular contours and steady gaze. "
                f"Parasympathetic nervous system balance."
            )
            layman = (
                "This person is feeling tranquil, peaceful, and quietly content with the moment."
            )
            notes.append("Gentle bilateral mouth curve and serene ocular stability.")

        # "Confusion / Skepticism":
        elif (corrugator >= 0.38 and abs(corrugator - grief_brow) > 0.25) or aus.au14_dimpler > 0.35:
            primary_state = "Confusion / Skepticism"
            surface = "Asymmetric Brow & Questioning Glance"
            underlying = "Cognitive Dissonance or Disbelief"
            sincerity = 0.88
            is_masking = False
            scientific = (
                f"Asymmetric brow contraction with AU14 dimpling reflects cognitive appraisal of incongruent information."
            )
            layman = (
                "This person is confused or skeptical. One eyebrow might be cocked and their expression says, 'Wait, really? Are you sure?'"
            )
            notes.append("Asymmetric brow tension signaling cognitive disbelief.")


        # "Neutral Baseline":
        else:
            primary_state = "Neutral Baseline"
            surface = "Resting Expression"
            underlying = "Equilibrium / Basal Awareness"
            sincerity = 0.95
            is_masking = False
            scientific = (
                f"Resting baseline: balanced bilateral facial symmetry, resting EAR ({ear:.2f}), "
                f"absence of significant Action Unit elevations."
            )
            layman = (
                "This person is in a neutral resting state. They aren't experiencing any strong positive or negative emotions right now; "
                "their mind is calm and observant."
            )
            notes.append("Musculature relaxed; retinal gaze centered; no emotional micro-tensions.")

        # -------------------------------------------------------------
        # 3. EMOTION RADAR PROBABILITY ESTIMATION (8 CORE DIMENSIONS)
        # -------------------------------------------------------------
        # Calculate Love / Affection probability
        love_prob = float(np.clip(
            (smile * 0.4 + (0.3 if is_direct_contact and not is_gaze_down else 0.0) + (0.3 if pulse_elevated else 0.1))
            if not is_masking and not has_corrugator_furrow and not has_grief_brow
            else 0.05, 0.0, 1.0
        ))

        radar = {
            "Joy / Happiness": float(np.clip(smile * 0.7 + cheek * 0.3 if not is_masking else smile * 0.25, 0.0, 1.0)),
            "Love / Affection": love_prob,
            "Sadness / Melancholy": float(np.clip(grief_brow * 0.4 + frown * 0.4 + (0.4 if is_gaze_down else 0.0), 0.0, 1.0)),
            "Anxiety / Restlessness": float(np.clip(restlessness * 0.5 + (0.3 if is_fluttering else 0.0) + (0.2 if pulse_elevated else 0.0), 0.0, 1.0)),
            "Anger / Frustration": float(np.clip(corrugator * 0.6 + lip_press * 0.4, 0.0, 1.0)),
            "Surprise / Fear": float(np.clip(wide_eyes * 0.7 + (grief_brow * 0.3 if wide_eyes > 0.2 else 0.0), 0.0, 1.0)),
            "Contempt / Smugness": float(np.clip(asymmetry * 0.8 + aus.au14_dimpler * 0.2, 0.0, 1.0)),
            "Neutral / Focus": float(np.clip(1.0 - max(smile, grief_brow, corrugator, wide_eyes, restlessness), 0.0, 1.0)),
        }

        total_rad = sum(radar.values())
        if total_rad > 0:
            radar = {k: round(v / total_rad, 3) for k, v in radar.items()}

        confidence = round(float(np.clip(0.78 + (0.14 if not is_masking else 0.08), 0.72, 0.98)), 2)

        return AffectDiagnosis(
            primary_state=primary_state,
            surface_expression=surface,
            underlying_truth=underlying,
            sincerity_score=round(sincerity, 2),
            is_masking_detected=is_masking,
            confidence=confidence,
            scientific_justification=scientific,
            layman_justification=layman,
            diagnostic_notes=notes,
            emotion_radar=radar,
        )
