"""Coverage, all-button safety and reproducible bounded layout sampling."""
import json
from pathlib import Path

import numpy as np
import pytest

from pressb.kinematics import PiperKinematics
from pressb.panel_randomization import (
    panel_camera_coverage, panel_randomization_settings, sample_panel_layout, validate_panel_layout,
)
from pressb.planning import button_centers


ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/scene.json").read_text())
COLLECTION_CFG = json.loads((ROOT / "configs/dataset_panel_randomized.json").read_text())
RANGE = COLLECTION_CFG["panel_randomization"]


@pytest.fixture(scope="module")
def kin():
    return PiperKinematics(ROOT / CFG["robot_urdf"], tip_offset=CFG["tip_offset"])


def randomized_cfg(**kwargs):
    return dict(CFG, panel_randomization={**RANGE, **kwargs})


@pytest.mark.parametrize("offset", [RANGE["min_offset_y_m"], RANGE["max_offset_y_m"]])
@pytest.mark.parametrize("offset_x", [RANGE["min_offset_x_m"], RANGE["max_offset_x_m"]])
@pytest.mark.parametrize("approach_delta", [-.003, 0., .003])
def test_complete_panel_and_all_buttons_safe_at_default_range_endpoints(kin, offset, offset_x, approach_delta):
    cfg = dict(COLLECTION_CFG, panel_offset_y_m=offset, panel_offset_x_m=offset_x,
               approach_distance=COLLECTION_CFG["approach_distance"] + approach_delta)
    proof = validate_panel_layout(kin, cfg)
    assert proof["all_buttons_reachable"]
    assert proof["reachable_floors"] == list(range(24, 36))
    assert proof["camera_coverage"]["minimum_margin_px"] >= 24
    assert proof["trajectory_validation"]["physics_samples_checked"] > 12000
    assert proof["trajectory_validation"]["conservative_clearances"]["fixed_camera_m"] >= .015
    reference = button_centers(CFG)
    shifted = button_centers(cfg)
    for floor in range(24, 36):
        np.testing.assert_allclose(shifted[floor] - reference[floor], [offset_x, offset, 0.], atol=1e-12)
    # Returned proofs cannot mutate a cached safety decision.
    proof["reachable_floors"].clear()
    assert len(validate_panel_layout(kin, cfg)["reachable_floors"]) == 12


def test_implicit_ranges_match_default_collection_config():
    implicit = panel_randomization_settings(dict(CFG, panel_randomization={"enabled": True}))
    assert implicit == panel_randomization_settings(COLLECTION_CFG)


def test_out_of_view_and_unsafe_camera_approach_are_rejected(kin):
    with pytest.raises(ValueError, match="fixed-camera coverage"):
        panel_camera_coverage(dict(CFG, panel_offset_y_m=.6))
    # The camera still sees the panel, but its close tabletop mount constrains
    # the arm's elbow path. Frustum-only validation must not accept this layout.
    assert panel_camera_coverage(dict(CFG, panel_offset_y_m=.04))["all_inside"]
    with pytest.raises(ValueError, match="fixed-camera clearance"):
        validate_panel_layout(kin, dict(CFG, panel_offset_y_m=.04))


def test_sampling_is_deterministic_separate_from_motion_rng_and_bounded(monkeypatch):
    checked = []

    def safe_layout(kin, cfg, margin_px):
        checked.append((cfg["panel_offset_x_m"], cfg["panel_offset_y_m"], margin_px))
        return {"all_buttons_reachable": True, "reachable_floors": list(range(24, 36))}

    monkeypatch.setattr("pressb.panel_randomization.validate_panel_layout", safe_layout)
    cfg = randomized_cfg()
    first, evidence = sample_panel_layout(None, cfg, 29, 179)
    repeat, same = sample_panel_layout(None, cfg, 29, 179)
    other, changed = sample_panel_layout(None, cfg, 29, 180)
    assert first == repeat and evidence == same
    assert first["panel_offset_y_m"] != other["panel_offset_y_m"]
    assert all(RANGE["min_offset_x_m"] <= x <= RANGE["max_offset_x_m"]
               and RANGE["min_offset_y_m"] <= y <= RANGE["max_offset_y_m"]
               and margin == RANGE["camera_margin_px"] for x, y, margin in checked)
    assert evidence["attempts"] == changed["attempts"] == 1
    assert "panel_offset_y_m" not in cfg
    assert len(checked) == 3


def test_rejection_sampling_has_finite_attempts_and_records_reasons(monkeypatch):
    calls = []

    def unsafe_then_safe(kin, cfg, margin_px):
        calls.append(cfg["panel_offset_y_m"])
        if len(calls) == 1:
            raise ValueError("deliberate unreachable floor 35")
        return {"all_buttons_reachable": True}

    monkeypatch.setattr("pressb.panel_randomization.validate_panel_layout", unsafe_then_safe)
    _, proof = sample_panel_layout(None, randomized_cfg(max_attempts=2), 24, 99)
    assert proof["attempts"] == 2
    assert len(proof["rejected_candidates"]) == 1
    assert proof["rejected_candidates"][0]["panel_offset_y_m"] == calls[0]
    assert "floor 35" in proof["rejected_candidates"][0]["reason"]

    calls.clear()
    with pytest.raises(ValueError, match="No safe panel layout after 1 attempts"):
        sample_panel_layout(None, randomized_cfg(max_attempts=1), 24, 99)
    assert len(calls) == 1


@pytest.mark.parametrize("settings", [
    {"min_offset_y_m": float("nan")}, {"max_offset_y_m": float("inf")},
    {"min_offset_y_m": .03, "max_offset_y_m": -.03}, {"enabled": "yes"},
    {"camera_margin_px": -1}, {"max_attempts": 0}, {"max_attempts": 1.2}, {"typo": 1},
    {"min_offset_x_m": .03, "max_offset_x_m": -.03}, {"min_offset_x_m": float("nan")},
])
def test_invalid_randomization_configuration_rejected(settings):
    with pytest.raises(ValueError):
        panel_randomization_settings(dict(CFG, panel_randomization=settings))


def test_disabled_layout_does_not_invoke_new_planning(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("legacy centered scene must not invoke layout sampling")

    monkeypatch.setattr("pressb.panel_randomization.validate_panel_layout", unexpected)
    cfg = dict(CFG, panel_randomization={"enabled": False})
    layout, proof = sample_panel_layout(None, cfg, 24, 0)
    assert layout == cfg
    assert proof == {"enabled": False, "panel_offset_y_m": 0., "panel_offset_x_m": 0.}
