"""Task-local home poses preserve the calibrated camera and reset contracts."""
from copy import deepcopy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from pressb.robots.piper.kinematics import PiperKinematics
from pressb.tasks.playground.backend import load_config as workspace_config


ROOT = Path(__file__).resolve().parents[1]
PRESET = ROOT / "configs/calibration/playground_20261113_appearance.json"
TASKS = ("cube_stack", "tabletop_cleanup")


def load(task, options=None):
    return importlib.import_module(f"pressb.tasks.{task}.configuration").load_config(options or {})


@pytest.mark.parametrize("task", TASKS)
def test_default_home_is_over_workspace_center_and_preserves_camera_calibration(task):
    before = PRESET.read_bytes()
    original = workspace_config({"scene_config": str(PRESET)})["scene"]
    cfg = load(task)
    scene, pose = cfg["scene"], cfg["initial_pose"]
    assert pose["mode"] == "workspace_center"
    initial = np.asarray(scene["initial_state_joint_rad_m"])
    kin = PiperKinematics(scene["robot_urdf"], tip_offset=.1358)
    actual = kin.fk(initial[:6])
    center = [np.mean(cfg["randomization"][axis]) for axis in ("x_range_m", "y_range_m")]
    np.testing.assert_allclose(actual[:2, 3], center, atol=1e-5)
    assert actual[2, 3]+scene["robot_base_height_above_table_m"] == pytest.approx(pose["height_above_table_m"], abs=1e-5)
    np.testing.assert_allclose(actual[:3, :3], pose["tool_rotation_base"], atol=1e-5)
    assert pose["min_joint_limit_margin_deg"] > 0
    assert initial[6] == pytest.approx(scene["gripper_max_width_m"])
    np.testing.assert_array_equal(initial[:6], scene["initial_joint_positions_rad"])
    assert initial[6] == scene["initial_gripper_width_m"]
    for key in ("up_camera", "wrist_camera", "wrist_camera_resolution", "overview_camera",
                "robot_base_xy", "robot_base_height_above_table_m", "table_height", "lighting", "materials"):
        assert scene[key] == original[key], key
    assert PRESET.read_bytes() == before


@pytest.mark.parametrize("task", TASKS)
def test_center_tracks_configured_region_rather_than_sampled_objects(task):
    cfg = load(task, {"randomization": {"x_range_m": [.25, .45], "y_range_m": [-.25, .17]}})
    np.testing.assert_allclose(cfg["initial_pose"]["center_xy_m"], [.35, -.04], atol=1e-12)
    np.testing.assert_allclose(cfg["initial_pose"]["tcp_actual_base_m"][:2], [.35, -.04], atol=1e-5)


@pytest.mark.parametrize("task", TASKS)
def test_reset_and_scene_authoring_use_the_same_centered_seven_coordinate_state(task):
    module = importlib.import_module(f"pressb.tasks.{task}.parallel")
    cls = module.CubeStackTask if task == "cube_stack" else module.TabletopCleanupTask
    plugin = cls({"num_envs": 2})
    expected = plugin.config["scene"]["initial_state_joint_rad_m"]
    for env_id, seed in ((0, 8), (1, 31)):
        spec = plugin.sample_reset(env_id, seed, {})
        assert spec["initial_state_joint_rad_m"] == expected
        assert spec["env_id"] == env_id
    scene = plugin.config["scene"]
    assert [*scene["initial_joint_positions_rad"], scene["initial_gripper_width_m"]] == expected


@pytest.mark.parametrize("task", TASKS)
def test_explicit_scene_joint_override_and_legacy_mode_are_preserved(task):
    original = workspace_config({"scene_config": str(PRESET)})["scene"]["initial_state_joint_rad_m"]
    custom = deepcopy(original)
    custom[0] += .01
    custom[6] = .05
    cfg = load(task, {"scene_overrides": {"initial_state_joint_rad_m": custom}})
    assert cfg["initial_pose"]["mode"] == "scene"
    assert cfg["scene"]["initial_state_joint_rad_m"] == custom
    assert [*cfg["scene"]["initial_joint_positions_rad"], cfg["scene"]["initial_gripper_width_m"]] == custom
    legacy = load(task, {"initial_pose": {"mode": "scene"}})
    assert legacy["scene"]["initial_state_joint_rad_m"] == original


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("options", [
    {"mode": "unknown"}, {"mode": "scene", "height_above_table_m": .16},
    {"height_above_table_m": -1.}, {"gripper_width_m": .101},
    {"radial_tilt_deg": float("nan")}, {"unexpected": True},
])
def test_invalid_centering_options_fail_on_cpu(task, options):
    with pytest.raises(ValueError):
        load(task, {"initial_pose": options})


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("layout", ("simulation", "real_layout"))
def test_active_presets_explicitly_select_centered_home(task, layout):
    node = json.loads((ROOT / "configs/rl_nodes" / f"{task}_{layout}.json").read_text())
    raw = node["backend"]["config"]
    assert raw["initial_pose"] == {"mode": "workspace_center"}
    assert "initial_state_joint_rad_m" not in raw.get("scene_overrides", {})
    assert load(task, raw)["initial_pose"]["mode"] == "workspace_center"
