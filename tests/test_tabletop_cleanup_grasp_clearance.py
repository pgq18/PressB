"""Continuous mesh sweeps, rather than a scalar bridge width, center grasps."""
from pathlib import Path

import numpy as np
import pytest

from pressb.tasks.tabletop_cleanup.geometry import asset_spec
from pressb.tasks.tabletop_cleanup.grasp_clearance import optimize_grasp_center, sweep_clearance

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "vendor/robot_lab/source/robot_lab/data/Robots/Agilex/PIPER/piper_description.urdf"


@pytest.fixture(scope="module")
def aligned():
    return optimize_grasp_center("gamepad", urdf_path=URDF, opening_m=.070)


def test_full_finger_sweep_accounts_for_stick_socket_not_just_bridge(aligned):
    before = aligned["before_clearance_m"]
    # A scalar 55 mm bridge estimate incorrectly predicts 7.5 mm both sides.
    assert .002 < before["negative"] < .0035
    assert before["positive"] == pytest.approx(.0075, abs=2e-6)
    assert aligned["limiting_parts"]["negative"].startswith("stick_socket")
    assert aligned["limiting_parts"]["positive"] == "center_bridge"


def test_centering_increases_worst_clearance_and_balances_sides(aligned):
    before, after = aligned["before_clearance_m"], aligned["after_clearance_m"]
    assert after["negative"] == pytest.approx(after["positive"], abs=1e-8)
    assert min(after.values()) > min(before.values())+.002
    assert aligned["center_shift_m"] == pytest.approx(-.002439, abs=2e-6)
    original = np.asarray(asset_spec("gamepad")["grasp_candidates"][0]["center_local_m"])
    np.testing.assert_allclose(np.asarray(aligned["center_local_m"])[[0,2]], original[[0,2]], atol=1e-12)
    np.testing.assert_allclose(np.asarray(aligned["tcp_local_m"])-aligned["center_local_m"], [0,0,-.004], atol=1e-12)
    assert aligned["opening_m"] == .070


def test_bound_limits_center_motion_and_still_improves_clearance():
    limited = optimize_grasp_center("gamepad", urdf_path=URDF, opening_m=.070,
                                    max_center_shift_m=.001)
    assert limited["center_shift_m"] == pytest.approx(-.001)
    assert min(limited["after_clearance_m"].values()) > min(limited["before_clearance_m"].values())
    assert limited["after_clearance_m"]["negative"] < limited["after_clearance_m"]["positive"]


def test_continuous_sweep_detects_overlap_at_lateral_offset(aligned):
    tcp = np.asarray(aligned["tcp_local_m"])
    tcp[1] += .006
    clearance = sweep_clearance("gamepad", urdf_path=URDF, opening_m=.070,
                                tcp_local_m=tcp, rotation_local=aligned["rotation_local"])
    assert clearance["negative"]["clearance_m"] == pytest.approx(0., abs=1e-9)


def test_shorter_and_longer_empty_approach_have_same_limiting_surfaces(aligned):
    # Both sweeps start at the contact TCP and extend well clear of the prop.
    longer = sweep_clearance("gamepad", urdf_path=URDF, opening_m=.070,
                             tcp_local_m=aligned["tcp_local_m"], rotation_local=aligned["rotation_local"],
                             approach_distance_m=.20)
    for side in ("negative", "positive"):
        assert longer[side]["clearance_m"] == pytest.approx(aligned["after_clearance_m"][side], abs=1e-7)
        assert longer[side]["part"] == aligned["limiting_parts"][side]


@pytest.mark.parametrize("opening_m", [.090, .100])
def test_sweep_respects_explicit_scene_opening_limit(aligned, opening_m):
    widened = optimize_grasp_center("gamepad", urdf_path=URDF, opening_m=opening_m,
                                    max_opening_m=.100)
    expected_gain = (opening_m - aligned["opening_m"]) / 2
    for side, clearance_m in widened["after_clearance_m"].items():
        assert clearance_m == pytest.approx(aligned["after_clearance_m"][side] + expected_gain,
                                            abs=1e-7)


def test_invalid_opening_and_rotation_fail_without_simulator(aligned):
    for opening in (.080, float("nan"), -.01):
        with pytest.raises(ValueError):
            optimize_grasp_center("gamepad", urdf_path=URDF, opening_m=opening)
    with pytest.raises(ValueError):
        optimize_grasp_center("gamepad", urdf_path=URDF, opening_m=.101, max_opening_m=.100)
    with pytest.raises(ValueError, match="proper rotation"):
        sweep_clearance("gamepad", urdf_path=URDF, opening_m=.070,
                        tcp_local_m=aligned["tcp_local_m"], rotation_local=np.ones((3,3)))
