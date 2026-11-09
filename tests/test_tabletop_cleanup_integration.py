"""Tabletop-cleanup examples retain the shared node and calibrated task contract."""
from copy import deepcopy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pressb.experiments.config import PROJECT_ROOT, load_experiment
from pressb.registry import resolve


def test_tabletop_cleanup_registry_discovery_is_lazy():
    code = """
from pressb.registry import resolve
import sys
assert resolve('task', 'tabletop_cleanup') == 'pressb.tasks.tabletop_cleanup.parallel:create_backend'
assert not any(name in sys.modules for name in ('torch', 'isaacsim'))
assert not any(name.startswith('pressb.tasks.') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True,
                   env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")})


def test_generic_smoke_exposes_tabletop_cleanup_without_isaac():
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/smoke_parallel_tasks.py"), "--help"],
        capture_output=True, text=True, check=True)
    assert "tabletop_cleanup" in result.stdout


def test_tabletop_cleanup_node_and_experiment_compile_to_matching_contracts():
    node = json.loads((PROJECT_ROOT / "configs/rl_nodes/tabletop_cleanup_simulation.json").read_text())
    compiled = load_experiment(PROJECT_ROOT / "configs/experiments/tabletop_cleanup_walloss_eval.json")
    simulation, inference, trainer = (compiled["nodes"][key] for key in ("simulation", "inference", "trainer"))
    assert compiled["factories"]["task"] == resolve("task", "tabletop_cleanup")
    assert node["backend"]["plugin"] == simulation["backend"]["plugin"] == "tabletop_cleanup"
    local, experiment = node["backend"]["config"], simulation["backend"]["config"]
    assert local["num_envs"] == experiment["num_envs"] == trainer["rollout"]["num_envs"] == 8
    assert local["horizon"] == experiment["horizon"] == trainer["rollout"]["execution_horizon"] == 32
    assert local["max_episode_seconds"] == experiment["max_episode_seconds"] == 120
    assert trainer["rollout"]["eval_episodes"] == 16
    assert trainer["rollout"]["max_transitions"] == 4096  # Allows all 16 to reach the 120 s limit.
    assert Path(experiment["scene_config"]) == PROJECT_ROOT / local["scene_config"]
    assert Path(experiment["scene_config"]).name == "playground_20261109_global_refined.json"
    for key in ("objects", "bin", "smoothing", "render_resolution", "randomization", "reward", "terminate_on_success"):
        assert local[key] == experiment[key]
    assert "scene_overrides" not in experiment
    assert "reset_render_frames" not in experiment
    assert node["server"] == simulation["server"]
    assert trainer["endpoints"]["simulation"] == "http://127.0.0.1:18424"
    assert trainer["endpoints"]["inference"] == "ws://127.0.0.1:18454"
    assert inference["backend"]["plugin"] == "walloss"
    assert inference["service"]["batch_size"] == 8


def _node_config(name):
    path = PROJECT_ROOT / "configs/rl_nodes" / f"tabletop_cleanup_{name}.json"
    return json.loads(path.read_text())["backend"]["config"]


@pytest.mark.parametrize("entrypoint", ["plugin_default", "simulation", "real_layout", "experiment"])
def test_example_samplers_keep_exactly_two_distinct_objects_and_fixed_bin(entrypoint):
    from pressb.tasks.tabletop_cleanup.configuration import load_config, sample_objects
    if entrypoint == "plugin_default":
        raw = {}
    elif entrypoint == "experiment":
        compiled = load_experiment(PROJECT_ROOT / "configs/experiments/tabletop_cleanup_walloss_eval.json")
        raw = compiled["nodes"]["simulation"]["backend"]["config"]
    else:
        raw = _node_config(entrypoint)
    config = load_config(raw)
    # The frozen box is independent of which entry point or random object pair
    # is selected. A new measured pose must migrate every example together.
    expected_bin = dict(center_xy=[.3396856, .21861745], yaw_deg=2.070158,
                        size_xyz_m=[.222, .148, .049], wall_thickness_m=.004)
    assert config["bin"] == config["scene"]["cleanup_bin"] == expected_bin
    fixed_bin = deepcopy(config["bin"])
    pairs = set()
    for seed in range(32):
        layout = sample_objects(config, seed % 8, seed)
        assert len(layout["active_names"]) == len(set(layout["active_names"])) == 2
        assert sum(layout["active_mask"]) == 2
        assert set(layout["objects"]) == set(layout["active_names"])
        assert set(layout["active_names"]) <= {"gamepad", "speaker", "lidar"}
        assert config["bin"] == config["scene"]["cleanup_bin"] == fixed_bin
        assert layout == sample_objects(config, seed % 8, seed)
        pairs.add(frozenset(layout["active_names"]))
    selected = config["randomization"]["fixed_subset"]
    expected_pairs = ([selected] if selected else
                      [("gamepad", "speaker"), ("gamepad", "lidar"), ("speaker", "lidar")])
    assert pairs == {frozenset(pair) for pair in expected_pairs}


def test_fixed_layout_keeps_two_objects_and_calibration_without_mutating_template():
    from pressb.tasks.tabletop_cleanup.configuration import load_config, sample_objects
    path = PROJECT_ROOT / "configs/calibration/playground_20261109_global_refined.json"
    original = path.read_bytes()
    scene = json.loads(original)
    expected_initial = (scene["initial_state_joint_rad_m"] if "initial_state_joint_rad_m" in scene else
                        [*scene["initial_joint_positions_rad"], scene["initial_gripper_width_m"]])
    for name in ("simulation", "real_layout"):
        raw = _node_config(name)
        config = load_config(raw)
        # A fixed capture may pin measured home feedback without modifying
        # the shared calibration's cameras or its template initial state.
        selected_initial = raw.get("scene_overrides", {}).get("initial_state_joint_rad_m", expected_initial)
        assert config["scene"]["initial_state_joint_rad_m"] == selected_initial
        for camera in ("up_camera", "wrist_camera"):
            assert config["scene"][camera] == scene[camera]
        assert config["reset_render_frames"] == scene["rendering"]["warmup_frames"]
        if name == "real_layout":
            assert raw["randomization"]["enabled"] is False
            assert len(set(raw["randomization"]["fixed_subset"])) == 2
            a, b = (sample_objects(config, 0, seed) for seed in (1, 41))
            assert a == b
            assert set(a["objects"]) == set(raw["randomization"]["fixed_subset"])
            for kind, pose in a["objects"].items():
                assert pose["center_xy"] == raw["objects"][kind]["center_xy"]
                assert pose["yaw_deg"] == raw["objects"][kind]["yaw_deg"]
    assert path.read_bytes() == original


@pytest.mark.parametrize("count", [1, 3])
def test_explicit_reset_from_example_cannot_change_required_object_count(count):
    from pressb.tasks.tabletop_cleanup.configuration import load_config, sample_objects
    config = load_config(_node_config("simulation"))
    raw = _node_config("real_layout")
    overrides = {kind: {"center_xy": spec["center_xy"], "yaw_deg": spec["yaw_deg"]}
                 for kind, spec in list(raw["objects"].items())[:count]}
    with pytest.raises(ValueError):
        sample_objects(config, 0, 42, overrides=overrides)
