"""Photometric updates preserve the registered scene and executable motions."""
from copy import deepcopy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "configs/calibration/playground_20261109_global_refined.json"
NEW = ROOT / "configs/calibration/playground_20261113_appearance.json"
APPEARANCE_FIELDS = {
    "materials", "lighting", "camera_color_response", "cleanup_materials", "robot_material", "reconstruction",
}


def without_appearance(scene):
    return {key: value for key, value in scene.items() if key not in APPEARANCE_FIELDS}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def load(task, preset):
    module = "backend" if task == "playground" else "configuration"
    loader = importlib.import_module(f"pressb.tasks.{task}.{module}").load_config
    return loader({"scene_config": str(preset)})


def test_new_template_changes_only_visual_properties_and_working_alias_is_exact():
    prior, current = (json.loads(path.read_text()) for path in (OLD, NEW))
    assert canonical(without_appearance(prior)) == canonical(without_appearance(current))
    assert prior["materials"] != current["materials"]
    assert prior["lighting"] != current["lighting"]
    assert current["camera_color_response"]
    assert (ROOT / "configs/piper_playground.json").read_bytes() == NEW.read_bytes()
    assert current["objects"] == []


@pytest.mark.parametrize("task", ["playground", "cube_stack", "tabletop_cleanup"])
def test_loaded_task_has_identical_geometry_cameras_physics_and_home(task):
    prior, current = (load(task, path) for path in (OLD, NEW))
    assert canonical(without_appearance(prior["scene"])) == canonical(without_appearance(current["scene"]))
    if task != "playground":
        for key in ("initial_pose", "randomization", "reward", "outcome", "terminate_on_success"):
            assert canonical(prior[key]) == canonical(current[key])


@pytest.mark.parametrize("task", ["cube_stack", "tabletop_cleanup"])
def test_sampling_unchanged_for_multiple_seeds_and_parallel_environment_ids(task):
    module = importlib.import_module(f"pressb.tasks.{task}.configuration")
    prior, current = (load(task, path) for path in (OLD, NEW))
    for env_id, seed in ((0, 0), (1, 17), (7, 99), (63, 8)):
        assert module.sample_objects(prior, env_id, seed) == module.sample_objects(current, env_id, seed)


@pytest.mark.parametrize("task", ["cube_stack", "tabletop_cleanup"])
@pytest.mark.parametrize("profile", ["legacy", "compact"])
def test_complete_action_plans_are_bitwise_identical_across_appearance_presets(task, profile):
    planner = importlib.import_module(f"pressb.tasks.{task}.planning")
    configs = [load(task, path) for path in (OLD, NEW)]
    if task == "cube_stack":
        plans = [planner.plan_stack(config["scene"], trajectory_profile=profile) for config in configs]
    else:
        plans = [planner.plan_cleanup(config, trajectory_profile=profile) for config in configs]
    before, after = plans
    assert before["phases"] == after["phases"]
    assert np.asarray(before["actions"], dtype="<f8").tobytes() == np.asarray(
        after["actions"], dtype="<f8").tobytes()


@pytest.mark.parametrize("task", ["cube_stack", "tabletop_cleanup"])
@pytest.mark.parametrize("layout", ["simulation", "real_layout"])
def test_current_task_entrypoints_pin_the_new_scene(task, layout):
    path = ROOT / "configs/rl_nodes" / f"{task}_{layout}.json"
    raw = json.loads(path.read_text())["backend"]["config"]
    assert ROOT / raw["scene_config"] == NEW
    assert raw["initial_pose"] == {"mode": "workspace_center"}
    plugin = importlib.import_module(f"pressb.tasks.{task}.configuration")
    resolved = plugin.load_config(deepcopy(raw))
    assert resolved["scene"]["camera_color_response"] == json.loads(NEW.read_text())["camera_color_response"]
