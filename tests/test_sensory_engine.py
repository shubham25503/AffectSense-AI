import os
import sys

# Ensure workspace root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import pytest
from engine.detector import SensoryPipeline



@pytest.fixture(scope="module")
def pipeline():
    pipe = SensoryPipeline()
    yield pipe
    pipe.close()


def test_pipeline_initialization(pipeline):
    assert pipeline is not None
    assert pipeline.detector is not None


def test_masked_sadness_detection(pipeline):
    path = os.path.join("sample_data", "masked_sadness.jpg")
    assert os.path.exists(path), f"Sample file not found: {path}"
    img = cv2.imread(path)
    res = pipeline.process_frame(img)

    assert res is not None, "Face should be detected in masked sadness image"
    assert res.face_detected is True

    # Validate that smile is present
    assert res.aus.au12_lip_corner_puller >= 0.35, "Mouth smile should be detected"

    # Validate that downward gaze or grief brow or low Duchenne is detected
    assert res.gaze.avg_pitch < 0.0 or res.aus.au1_inner_brow_raiser > 0.0, "Subconscious sadness signals should be active"

    # Validate that sincerity is low / masking is flagged
    assert res.affect.is_masking_detected is True, "Masking should be detected for fake/sad smile"
    assert res.affect.sincerity_score < 0.60, f"Sincerity should be low, got {res.affect.sincerity_score}"
    assert "Masked" in res.affect.primary_state or "Sadness" in res.affect.primary_state or "Forced" in res.affect.primary_state


def test_genuine_joy_detection(pipeline):
    path = os.path.join("sample_data", "genuine_joy.jpg")
    assert os.path.exists(path), f"Sample file not found: {path}"
    img = cv2.imread(path)
    res = pipeline.process_frame(img)

    assert res is not None, "Face should be detected in genuine joy image"
    assert res.face_detected is True
    assert res.aus.au12_lip_corner_puller >= 0.45, "Mouth smile should be high"
    assert res.aus.duchenne_coherence >= 0.50, f"Duchenne coherence should be elevated, got {res.aus.duchenne_coherence}"
    assert res.affect.sincerity_score >= 0.70, f"Sincerity should be high, got {res.affect.sincerity_score}"
    assert res.affect.is_masking_detected is False, "Masking should NOT be detected for genuine smile"


def test_in_love_detection(pipeline):
    path = os.path.join("sample_data", "in_love.jpg")
    assert os.path.exists(path), f"Sample file not found: {path}"
    img = cv2.imread(path)
    res = pipeline.process_frame(img)

    assert res is not None, "Face should be detected in in_love image"
    assert res.face_detected is True
    assert "Love" in res.affect.primary_state or "Affection" in res.affect.primary_state
    assert res.affect.sincerity_score >= 0.85
    assert res.affect.is_masking_detected is False
    assert len(res.affect.scientific_justification) > 10
    assert len(res.affect.layman_justification) > 10



def test_neutral_focus_detection(pipeline):
    path = os.path.join("sample_data", "neutral_focus.jpg")
    assert os.path.exists(path), f"Sample file not found: {path}"
    img = cv2.imread(path)
    res = pipeline.process_frame(img)

    assert res is not None
    assert res.face_detected is True
    # Smile should be low
    assert res.aus.au12_lip_corner_puller < 0.35
    assert res.affect.is_masking_detected is False


def test_hud_rendering(pipeline):
    path = os.path.join("sample_data", "masked_sadness.jpg")
    img = cv2.imread(path)
    res = pipeline.process_frame(img)
    annotated = pipeline.draw_hud(img, res)

    assert annotated.shape == img.shape
    assert annotated.dtype == img.dtype


if __name__ == "__main__":
    pipe = SensoryPipeline()
    print("Running manual test on masked sadness...")
    img1 = cv2.imread("sample_data/masked_sadness.jpg")
    r1 = pipe.process_frame(img1)
    print(f"Masked Sadness -> State: {r1.affect.primary_state}, Sincerity: {r1.affect.sincerity_score*100}%, Masking: {r1.affect.is_masking_detected}")
    print(f"Notes: {r1.affect.diagnostic_notes}")

    print("\nRunning manual test on genuine joy...")
    img2 = cv2.imread("sample_data/genuine_joy.jpg")
    r2 = pipe.process_frame(img2)
    print(f"Genuine Joy -> State: {r2.affect.primary_state}, Sincerity: {r2.affect.sincerity_score*100}%, Masking: {r2.affect.is_masking_detected}")
    print(f"Notes: {r2.affect.diagnostic_notes}")

    print("\nRunning manual test on neutral focus...")
    img3 = cv2.imread("sample_data/neutral_focus.jpg")
    r3 = pipe.process_frame(img3)
    print(f"Neutral -> State: {r3.affect.primary_state}, Sincerity: {r3.affect.sincerity_score*100}%, Masking: {r3.affect.is_masking_detected}")

    pipe.close()
    print("\nALL SENSORY TESTS COMPLETED SUCCESSFULLY!")
