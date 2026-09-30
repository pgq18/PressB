"""Safety and causal-label checks for single-floor collection."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pressb.dataset_planning import ee_pose_base, make_episode_plan, sample_state_action, validate_episode_plan
from pressb.kinematics import PiperKinematics
from pressb.planning import make_plan

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/scene.json").read_text())


@pytest.fixture(scope="module")
def kin():
    return PiperKinematics(ROOT / CFG["robot_urdf"], tip_offset=CFG["tip_offset"])


@pytest.mark.parametrize("stride", [12, 6, 4])
def test_state_is_gripper_tip_and_action_is_next_control_endpoint(kin, stride):
    q = np.tile(CFG["home_q"], (30, 1))
    q[:, 1] += np.linspace(0., .08, 30)
    plan = SimpleNamespace(q=q)
    state, action = sample_state_action(kin, plan, q[0], [.0041, -.0039], 0, stride)
    expected = kin.link_transforms(q[0])["link6"] @ [0., 0., .1358, 1.]
    np.testing.assert_allclose(state[:3], expected[:3], atol=1e-12)
    np.testing.assert_allclose(action, ee_pose_base(kin, q[stride]), atol=1e-12)
    assert not np.allclose(action[:3], ee_pose_base(kin, q[0])[:3])
    assert np.isclose(np.linalg.norm(state[3:7]), 1.)
    assert np.isclose(state[7], .008)
    _, terminal = sample_state_action(kin, plan, q[-1], [.004, -.004], 29, stride)
    np.testing.assert_allclose(terminal, ee_pose_base(kin, q[-1]), atol=1e-12)


def test_seeded_episodes_keep_home_and_succeed_safety_check(kin):
    first = make_episode_plan(kin, CFG, 35, 17)
    repeat = make_episode_plan(kin, CFG, 35, 17)
    other = make_episode_plan(kin, CFG, 35, 18)
    np.testing.assert_array_equal(first.q, repeat.q)
    assert first.metadata == repeat.metadata
    assert first.metadata["joint_trajectory_sha256"] != other.metadata["joint_trajectory_sha256"]
    assert set(first.floor) == {35}
    np.testing.assert_allclose(first.q[[0, -1]], [CFG["home_q"], CFG["home_q"]], atol=1e-12)
    assert first.metadata["validation"]["conservative_clearances"]["fixed_camera_m"] >= .015
    damaged = SimpleNamespace(**{k: getattr(first, k) for k in ("q", "phase", "target_tip")})
    damaged.q = first.q.copy()
    damaged.q[len(damaged.q) // 2, 0] = kin.upper[0] + .1
    with pytest.raises(ValueError, match="joint limits"):
        validate_episode_plan(kin, damaged, CFG)


def test_disabling_panel_randomization_preserves_historical_seeded_trajectory(kin):
    cfg = dict(CFG, panel_randomization={"enabled": False}, panel_offset_y_m=0.)
    floor, seed = 35, 17
    # Reproduce the pre-randomization caller's established random draws. This
    # protects dataset reproducibility without machine-specific float hashes.
    rng = np.random.default_rng(np.random.SeedSequence([seed, floor]))
    legacy_cfg = dict(CFG, sequence=[floor],
                      approach_distance=CFG["approach_distance"] + rng.uniform(-.003, .003),
                      joint_speed=CFG["joint_speed"] * rng.uniform(.88, 1.0),
                      press_duration=rng.uniform(1.30, 1.65),
                      retract_duration=rng.uniform(1.30, 1.65),
                      dwell_duration=rng.uniform(.32, .46),
                      home_hold_duration=rng.uniform(.60, .75))
    legacy = make_plan(kin, legacy_cfg)
    actual = make_episode_plan(kin, cfg, floor, seed)
    np.testing.assert_array_equal(actual.q, legacy.q)
    np.testing.assert_array_equal(actual.target_tip, legacy.target_tip)
    assert actual.metadata["panel_offset_y_m"] == 0.
    assert not actual.metadata["scene_randomized"]


def test_randomized_episode_uses_offset_in_commanded_targets_and_metadata(kin):
    cfg = dict(CFG, panel_randomization={"enabled": True, "min_offset_y_m": -.02,
                                        "max_offset_y_m": .02})
    plan = make_episode_plan(kin, cfg, 35, 19)
    offset = plan.metadata["panel_offset_y_m"]
    assert -.02 <= offset <= .02
    assert plan.metadata["scene_randomized"]
    assert plan.metadata["variation"]["panel_offset_y_m"] == offset
    assert plan.metadata["panel_randomization"]["panel_offset_y_m"] == offset
    offset_x = plan.metadata["panel_offset_x_m"]
    assert -.01 <= offset_x <= .01
    assert plan.metadata["variation"]["panel_offset_x_m"] == offset_x
    assert plan.metadata["panel_randomization"]["panel_offset_x_m"] == offset_x
    assert plan.metadata["panel_randomization"]["validation"]["reachable_floors"] == list(range(24, 36))
    hold = plan.phase == "hold"
    np.testing.assert_allclose(plan.target_tip[hold, 1], offset - CFG["button_column_y"], atol=1e-12)
    np.testing.assert_allclose(plan.target_tip[hold, 0], CFG["button_face_x"] + offset_x + CFG["press_depth"], atol=1e-12)
    np.testing.assert_allclose(plan.q[[0, -1]], [CFG["home_q"], CFG["home_q"]], atol=1e-12)
