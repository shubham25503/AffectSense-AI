# The Research Behind AffectSense / Letstry

## Plain-language summary

Letstry is an **experimental facial-cue analysis system**. It uses a camera
frame to find a face, measure visible changes around the eyes, brows and mouth,
and—when it has video—measure change over time such as blinks and skin-colour
variation.

It combines several research traditions:

```text
facial anatomy + FACS movement coding + computer vision
  + eye/blink measurement + video signal processing
  = visible-cue measurements and heuristic interpretations
```

It is not a device that can read private thoughts, diagnose mental illness, or
reliably decide whether a person is lying. That difference is central to the
science.

---

## 1. What the application actually does

| Stage | What happens | Example |
|---|---|---|
| Detect | It locates one or more faces and follows them over video frames. | Two people in frame receive separate tracks. |
| Landmark | MediaPipe estimates points around lips, lids, brows, and irises. | The system gets top and bottom eyelid positions. |
| Measure | Letstry calculates mouth opening, eye opening, brow geometry, and left/right differences. | Raised mouth corners produce a larger AU12-style score. |
| Track time | It keeps short histories of eye position, eye openness, and forehead colour. | Low eye-opening values followed by reopening count as a blink. |
| Classify | Its own hand-written rules combine the measurements into a label. | A mouth-smile score with low estimated eye involvement may yield “forced smile.” |

The relevant code is:

- [detector.py](engine/detector.py) — face landmarking and the full pipeline
- [micro_expressions.py](engine/micro_expressions.py) — AU-style scores
- [gaze_tracker.py](engine/gaze_tracker.py) — iris/eye-landmark geometry
- [blink_detector.py](engine/blink_detector.py) — temporal blink estimates
- [rppg_pulse.py](engine/rppg_pulse.py) — tentative camera pulse estimate
- [sincerity_classifier.py](engine/sincerity_classifier.py) — final rule-based labels

---

## 2. The researchers and their contributions

### Guillaume Duchenne de Boulogne (1806–1875)

Duchenne was a French neurologist who investigated how facial muscles create
visible movement. His work helped establish the idea that a smile is not only
about the mouth: muscles around the eyes can move too.

The name **Duchenne smile** was later used for a pattern that includes raised lip
corners together with eye/cheek involvement. This is a *visible pattern*, not
proof that a person feels joy.

**Example:** Person A raises their mouth corners and cheeks, so the eye area
narrows. Person B makes a posed ID-photo smile with mainly the mouth. The
patterns can differ, but neither photo alone tells us why either person smiled.

### Charles Darwin (1809–1882)

In *The Expression of the Emotions in Man and Animals* (1872), Darwin treated
expression as observable behaviour with a possible evolutionary history. His
work helped inspire systematic study of faces. It does **not** provide a
universal dictionary saying that one face always equals one emotion.

### Carl-Herman Hjortsjö (1914–1993)

Hjortsjö, a Swedish anatomist, mapped visible facial movement to muscle groups.
His anatomical work was important groundwork for FACS.

### Paul Ekman, Wallace Friesen, and Joseph Hager

Ekman and Friesen published the first **Facial Action Coding System** (FACS) in
1978; Ekman, Friesen, and Hager updated it in 2002. FACS lets researchers
describe what a face visibly does in an anatomical language called **Action
Units** (AUs).

This distinction matters:

```text
Observation: AU12 (lip-corner puller) is visible.
Interpretation: “The person is happy.”
```

The first statement describes motion. The second requires context and can be
wrong. [FACS history](https://www.paulekman.com/facial-action-coding-system/)
and [academic overview](https://pmc.ncbi.nlm.nih.gov/articles/PMC3008166/).

### Ernest Haggard and Kenneth Isaacs

Haggard and Isaacs described very brief “micromomentary” facial events while
studying psychotherapy film in 1966. Later work by Ekman and Friesen made
short-lived facial behaviour widely known as a research topic.

Their contribution explains why real micro-expression research needs video:
the event must be observed beginning, peaking, and disappearing.
[Haggard & Isaacs (1966)](https://cir.nii.ac.jp/crid/1363388843349812992).

### Google MediaPipe researchers

Valentin Bazarevsky, Yury Kartynnik, Andrey Vakunov, Karthik Raveendran,
Matthias Grundmann, and collaborators developed efficient modern components for
mobile face detection and approximate 3D face geometry. Letstry uses the
MediaPipe Face Landmarker model to obtain its face mesh and blendshape outputs.

The Face Mesh work estimates an **approximate 3D mesh from a single camera
image**, originally for applications such as augmented reality.
[Paper](https://arxiv.org/abs/1907.06724).

### Tereza Soukupová and Jan Čech

Their landmark-based blink-detection work popularised a simple **Eye Aspect
Ratio** (EAR) approach. An open eye has more height relative to width; a closed
eye has much less.

**Example:**

```text
open eye:   height 12 px / width 40 px = EAR 0.30
closed eye: height  3 px / width 40 px = EAR 0.075
```

Letstry uses the same basic idea: values below a threshold for sufficient frames
are interpreted as a blink. [Paper](https://cmp.felk.cvut.cz/ftp/articles/cech/Soukupova-TR-2016-05.pdf).

### Wim Verkruysse, Lars Svaasand, and J. Stuart Nelson

Their 2008 work demonstrated **remote photoplethysmography** (rPPG): a camera
can sometimes observe tiny, pulse-related skin-colour changes under ambient
light. The green channel often has a relatively strong signal due to
haemoglobin absorption.

Letstry samples the forehead’s green channel, filters the time series, and
chooses a strong repeating frequency as a tentative BPM estimate.

**Example:** A repeating frequency of 1.2 Hz corresponds to approximately
`1.2 × 60 = 72 BPM`. [Study](https://opg.optica.org/oe/fulltext.cfm?uri=oe-16-26-21434).

---

## 3. The FACS-style signals shown in Letstry

| Score | Usual facial-movement description | Careful observation |
|---|---|---|
| AU1 | Inner brow raiser | The inner brow appears higher. |
| AU4 | Brow lowerer | The brows pull down/together. |
| AU5 | Upper lid raiser | The eye aperture appears larger. |
| AU6 | Cheek raiser | Cheeks lift; skin near the eye changes. |
| AU7 | Lid tightener | Eyelids appear narrowed/tightened. |
| AU12 | Lip-corner puller | Mouth corners pull upward. |
| AU14 | Dimpler / asymmetric pull | One side of the mouth may be more active. |
| AU15 | Lip-corner depressor | Mouth corners pull downward. |
| AU24 | Lip pressor | Lips are compressed together. |

Letstry is not performing certified manual FACS coding. Its AU-like numbers are
approximations derived from MediaPipe blendshapes and landmark geometry.

**Example:** A high AU4-style score means the brow appears lowered. It could
occur while concentrating, in sunlight, during pain, anger, confusion, or as a
person’s usual facial posture. The signal alone does not identify the cause.

---

## 4. Duchenne “coherence” in this project

Letstry’s “Duchenne coherence” asks a simplified question:

```text
Is there an AU12-like mouth smile?
AND
Is there AU6/AU7-like eye/cheek activity or reduced eye aperture?
```

**Example:**

- A laughing person has raised mouth corners, lifted cheeks, and narrowed eyes:
  the app may report high coherence.
- A person holds a polite smile with relatively wide eyes: it may report lower
  coherence.

This can be a useful description of the two visible patterns. It cannot justify
the fixed rule “eye crinkles mean genuine emotion; no eye crinkles mean fake.”
Anatomy, age, lighting, glasses, makeup, voluntary control, and social setting
all affect appearance. Research by Ekman, Richard Davidson, and Friesen studied
Duchenne and non-Duchenne smiles at a group level; it does not make a personal
truth detector. [Study](https://www.paulekman.com/wp-content/uploads/2013/07/The-Duchenne-Smile-Emotional-Expression-And-Brain-Physiolog.pdf).

---

## 5. Micro-expressions: a photo cannot prove one

A micro-expression is fundamentally a **time event**. A still photo can show
a facial configuration, but cannot establish:

- how long it lasted;
- whether it was suppressed;
- what happened before or after it; or
- whether it was involuntary.

**Example:** In a video, a brow movement lasting 150 ms before a smile can be
inspected frame by frame. In a single selfie, it is impossible to know whether
the brow movement lasted 150 ms, ten seconds, or never occurred.

Letstry retains histories for gaze and blinking, but it does not contain a
validated, high-frame-rate micro-expression model trained on a labelled
micro-expression dataset. In this project, “micro-expression” should mean
“small visible facial cue,” not a verified scientific finding.

---

## 6. Gaze: iris approximation, not retinal tracking

The project measures visible **iris/eye landmarks**. A standard RGB webcam
cannot see the retina, so “retinal gaze tracking” is not scientifically
accurate. A better description is:

> Webcam-based iris/eye-landmark gaze approximation.

The code estimates gaze from iris position relative to eye corners and lids,
then combines it with MediaPipe eye-look blendshapes.

**Example:** If the iris appears nearer the lower eyelid, the app may label the
estimated pitch as downward. But head rotation, camera position, glasses,
resolution, eyelid occlusion, lighting, and distance all change the result.

Gaze research treats calibration and head pose as central measurement problems.
[Survey](https://www.sciencedirect.com/science/article/abs/pii/S0031320322004241)

The responsible result wording is: “In this camera view, eye geometry is
consistent with a downward-looking appearance.” It is not: “This proves shame,
guilt, sadness, or deception.”

---

## 7. Blink and pulse limits

### Blinks

The app can reasonably report that it observed eye closures under its camera
conditions. It cannot reliably infer anxiety, stress, or lying from a blink
rate. Blinking varies with dry eyes, fatigue, screen use, lenses, medication,
lighting, conversation, and tracking error.

### rPPG pulse

Camera pulse sensing is sensitive to head movement, speaking, shadows,
auto-exposure, skin reflectance, compression, frame-rate changes, and the size
of the detected face. Letstry’s rPPG output is an experimental signal estimate,
not a medical measurement or a stress test.

Do not use it to diagnose a heart condition, emotional state, intoxication, or
mental-health condition.

---

## 8. Measurement versus the app’s final labels

The final labels do not come directly from FACS, MediaPipe, blink research, or
the rPPG paper. They are Letstry’s own rules in
[`engine/sincerity_classifier.py`](engine/sincerity_classifier.py).

| App label | Measurements it may combine | Scientifically careful wording |
|---|---|---|
| “Authentic joy” | Raised mouth corners and estimated eye/cheek involvement | “The visible pattern resembles a Duchenne-style smile.” |
| “Masked sadness” | Smile-like mouth plus downward-looking geometry or inner-brow signal | “Cues are mixed; do not infer hidden sadness without context.” |
| “Anxiety” | High estimated eye movement or blink count | “This clip has more measured eye movement/blinks under this setup.” |
| “Contempt” | Left/right smile difference | “The detected mouth movement is asymmetric.” |

This makes Letstry a **heuristic classifier**. It is valuable for demonstration,
visualisation, and generating research questions. It would need independently
collected data, predefined outcomes, diverse participants, error analysis, and
external validation before being used as an emotion-recognition tool.

---

## 9. Why emotion claims need context

Lisa Feldman Barrett, Ralph Adolphs, Stacy Marsella, Aleix M. Martinez, and
Seth D. Pollak reviewed more than 1,000 findings on facial movement and emotion.
Their conclusion: facial movements convey information, but a configuration does
not have one reliable, universal emotional meaning. Meaning varies with person,
culture, situation, and context.

A smile may occur with happiness, politeness, embarrassment, nervousness,
appeasement, sarcasm, or a camera pose. A scowl may occur with anger,
concentration, sunlight, pain, or confusion.

Read the review: [*Emotional Expressions Reconsidered* (2019)](https://www.psychologicalscience.org/journals/pspi/1529100619832930/).

For that reason, do not use Letstry as the sole basis for decisions about:

- honesty, sincerity, deception, or intent;
- depression, anxiety, grief, romance, guilt, or shame;
- hiring, education, policing, insurance, healthcare, or safety; or
- an assessment of anyone who has not meaningfully consented.

---

## 10. The most accurate one-sentence description

> AffectSense / Letstry is an experimental webcam tool that visualises
> landmark-based facial movement, eye openness, rough eye-direction estimates,
> blink dynamics, and tentative video pulse signals; its emotion labels are
> heuristic hypotheses, not verified readings of a person’s feelings or
> truthfulness.

---

## Selected reading

1. [Facial Action Coding System — history and current manual](https://www.paulekman.com/facial-action-coding-system/)
2. [Ekman, Davidson & Friesen — Duchenne smile study](https://www.paulekman.com/wp-content/uploads/2013/07/The-Duchenne-Smile-Emotional-Expression-And-Brain-Physiolog.pdf)
3. [Haggard & Isaacs (1966) — micromomentary facial expressions](https://cir.nii.ac.jp/crid/1363388843349812992)
4. [Bazarevsky et al. — MediaPipe Face Mesh](https://arxiv.org/abs/1907.06724)
5. [Soukupová & Čech — blink detection from landmarks](https://cmp.felk.cvut.cz/ftp/articles/cech/Soukupova-TR-2016-05.pdf)
6. [Verkruysse, Svaasand & Nelson — remote PPG](https://opg.optica.org/oe/fulltext.cfm?uri=oe-16-26-21434)
7. [Barrett et al. — limits of inferring emotion from faces](https://www.psychologicalscience.org/journals/pspi/1529100619832930/)
