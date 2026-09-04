# AffectSense AI: Human Emotion & Retinal Gaze Intelligence

## License

Copyright 2026 Shubham25503. This repository's original source code is licensed
under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party attribution.

Before redistributing this project, confirm that you have permission to
redistribute all sample images, models, and other non-code assets.

> **Beyond Superficial Face Expressions**: Detect authentic joy, masked sadness, forced smiling, and emotional suppression using **3D retinal/iris gaze tracking**, **Duchenne Action Unit coherence (AU6 vs AU12)**, **Darwinian grief muscle tension (AU1)**, and **contactless pulse sensing (rPPG)**.

---

## 🎯 The Core Philosophy
A conventional emotion model looks at a smiling mouth and classifies the person as "Happy". But smiling is frequently used to mask sadness, depression, anxiety, or social obligation.

**AffectSense AI** senses the subconscious human truth by tracking:
1. **Retinal Gaze Vectors (Pitch & Yaw)**: When people smile while masking dejection or grief, their retinal gaze involuntarily deflects downward towards the floor ($<-5^\circ$) or averts sideways.
2. **Duchenne Coherence (AU6/AU7 vs. AU12)**: Authentic joy co-activates the *Orbicularis Oculi* (outer eye crinkle / crow's feet) with the lips. In a fake or masked smile, the lips stretch while the eyes remain wide, blank, and disconnected.
3. **The Darwinian 'Grief Muscle' (AU1)**: Involuntary elevation of the inner eyebrows (*Frontalis Medialis*) creates micro-tensions even while the mouth is smiling.
4. **Ocular Micro-Saccades & Blink Dynamics**: Restless saccades, elevated blink rates (>30 BPM), and blink flutters reveal high internal anxiety and cognitive conflict.
5. **Contactless Pulse (rPPG)**: Estimates heart rate fluctuations from microscopic green-channel skin perfusion on forehead and cheek capillaries.

---

## 📚 16 Detectable States Catalog

### 1. Authentic Positive Senses
- **Authentic Joy (Duchenne Smile)**: Co-activated AU12 + AU6/AU7 with direct, steady gaze.
- **Playful Amusement / Delight**: High bilateral eye squint, open mouth, dynamic micro-saccades.
- **Serene Contentment**: Soft mouth curve, relaxed eyelids, steady resting pulse.

### 2. Masked, Forced & Deceptive States *(Core Feature)*
- **Masked Sadness ("Smiling Melancholy" / Smiling Depression)**: Mouth smiles, but retinal gaze vector is tilted downward, Duchenne orbital narrowing is absent, and inner brow grief tension is active.
- **Forced / Pan Am Smile (Social Courtesy)**: Mouth smile without eye engagement; eyes remain wide and emotionally detached.
- **Masked Anxiety / Nervous Placation**: Polite smile paired with blink flutter bursts (>35 BPM) and saccadic restlessness.
- **Contempt / Smug Disdain**: Asymmetric unilateral lip pull (unilateral AU12/AU14) with tilted/averted gaze.
- **Suppressed Frustration**: Outward smile masking *Corrugator* brow furrows (AU4) and tight lips (AU24).

### 3. Vulnerable, Inward & Affective States
- **Genuine Sadness / Grief**: Grief brow (AU1) + lip depression (AU15) + downward gaze aversion.
- **Shame / Dejection / Guilt**: Downward and sideways retinal deflection, avoiding eye contact.
- **Anxiety & Cognitive Restlessness**: Erratic micro-saccades, rapid blinks, scanning pupils.
- **Suppressed Anger / Resentment**: Furrowed brow (AU4) + pressed lips (AU24) + rigid gaze lock.
- **Fear / Panic**: Wide eye aperture (AU5) + raised brows + accelerated pulse.
- **Astonishment / Surprise**: Wide eyes + eyebrow elevation + dropped jaw.

### 4. Cognitive & Basal States
- **Deep Concentration / Inward Thought**: Narrowed eye aperture, micro-fixation, low blink rate.
- **Neutral Baseline**: Balanced symmetry, relaxed facial muscles, resting pulse.

---

## 🚀 Quickstart

### 1. Launch the Real-Time OpenCV HUD (Camera or Image)
Run the native HUD with your webcam:
```bash
python3 live_sense.py
```

Run against a static portrait image:
```bash
python3 live_sense.py --source sample_data/masked_sadness.jpg
python3 live_sense.py --source sample_data/genuine_joy.jpg
python3 live_sense.py --source sample_data/neutral_focus.jpg
```

**HUD Controls**:
- `Q` or `ESC`: Quit
- `M`: Toggle facial landmark wireframe
- `R`: Toggle 3D retinal gaze vector rays
- `S`: Save snapshot image and JSON biometric telemetry report

---

### 2. Launch the Modern Web Dashboard
Launch the interactive Streamlit dashboard:
```bash
streamlit run app.py
```
**Web Features**:
- **Photo Inspector**: 1-click test benchmarks (*Masked Sadness*, *Genuine Joy*, *Inward Focus*) or upload any custom photo.
- **Live Webcam Mode**: In-browser camera capture with instant biometric diagnosis.
- **Video Timeline Analyzer**: Upload video clips to chart sincerity and micro-expression leakage over time.
- **Data Export**: Export complete forensic JSON telemetry reports.

---

## 🧪 Running Automated Tests
Run the comprehensive test suite:
```bash
pytest tests/test_sensory_engine.py -v
```

---

## 📂 Project Structure
```
├── engine/
│   ├── __init__.py
│   ├── detector.py             # Master SensoryPipeline & HUD renderer
│   ├── gaze_tracker.py         # 3D Iris/Retina gaze pitch, yaw, EAR, restlessness
│   ├── micro_expressions.py    # FACS Action Units (AU1, AU4, AU6, AU7, AU12, etc.)
│   ├── blink_detector.py       # Blink rate, flutter bursts, and closure duration
│   ├── rppg_pulse.py           # Remote photoplethysmography pulse sensor
│   └── sincerity_classifier.py # Affect fusion, sincerity index, 16-state classifier
├── sample_data/
│   ├── masked_sadness.jpg      # Forced smile with sad downward gaze
│   ├── genuine_joy.jpg         # Radiant Duchenne smile
│   └── neutral_focus.jpg       # Calm inward concentration
├── tests/
│   └── test_sensory_engine.py  # Automated verification suite
├── live_sense.py               # Standalone OpenCV HUD application
├── app.py                      # Modern Streamlit Web Studio
├── requirements.txt            # Package dependencies
└── README.md                   # System documentation
```
