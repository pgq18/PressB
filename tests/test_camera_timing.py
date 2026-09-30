"""Visual synchrony evidence must use the actual collection frame rate."""
from fractions import Fraction
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("av")
pytest.importorskip("cv2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_camera_sync import feedback_pixel_mask, summarize_light, validate_video_rate


@pytest.mark.parametrize("fps", [10, 20, 30, 60])
def test_camera_audit_accepts_matching_capture_and_video_rates(fps):
    stream = SimpleNamespace(average_rate=Fraction(fps, 1))
    assert validate_video_rate(stream, {"fps": fps}, {"fps": fps}) == fps


def test_thirty_hz_video_cannot_use_old_ten_hz_episode_or_collection_metadata():
    stream = SimpleNamespace(average_rate=Fraction(30, 1))
    with pytest.raises(ValueError, match="Episode fps"):
        validate_video_rate(stream, {"fps": 10}, {"fps": 30})
    with pytest.raises(ValueError, match="Video fps"):
        validate_video_rate(stream, {"fps": 10}, {"fps": 10})


@pytest.mark.parametrize("rate", [None, Fraction(30000, 1001)])
def test_missing_or_nearby_video_rate_cannot_claim_thirty_hz(rate):
    with pytest.raises(ValueError, match="Video fps"):
        validate_video_rate(SimpleNamespace(average_rate=rate), {"fps": 30}, {"fps": 30})


def test_amber_edge_stays_on_across_red_green_ratio_boundary():
    # Actual pilot F34 lit-edge colors; legacy R>1.25G rejects the first three.
    lit = np.array([[[229, 187, 64], [219, 181, 70], [218, 181, 72], [220, 170, 60]]], dtype=np.uint8)
    assert feedback_pixel_mask(lit).all()
    off = np.array([[[40, 42, 45], [225, 225, 225], [250, 225, 190],
                     [255, 0, 0], [0, 255, 0], [0, 0, 255], [80, 55, 10]]], dtype=np.uint8)
    assert not feedback_pixel_mask(off).any()


def light_samples(press=40, release=61):
    # 30 Hz labels: on at 40/30 s, off at 61/30 s; amber and white have equal V.
    on = np.broadcast_to(np.array([219, 181, 70], dtype=np.uint8), (10, 10, 3))
    off = np.full((10, 10, 3), 219, dtype=np.uint8)
    return [{"orange_pixels": int(feedback_pixel_mask(on if press <= i < release else off).sum()),
             "time_s": i/30.} for i in range(100)]


def test_thirty_hz_amber_edge_reports_exact_press_and_release():
    summary = summarize_light(light_samples(), 40, 60)
    assert summary["aligned"]
    assert summary["visible_first_lit_frame"] == 40
    assert summary["visible_first_unlit_frame"] == 61
    assert summary["first_unlit_orange_pixels"] == 0


@pytest.mark.parametrize("press,release", [(39, 61), (41, 61), (40, 60), (40, 62)])
def test_one_thirty_hz_frame_of_press_or_release_error_is_rejected(press, release):
    summary = summarize_light(light_samples(press, release), 40, 60)
    assert summary["sufficient_visibility"]
    assert summary["aligned"] is False
