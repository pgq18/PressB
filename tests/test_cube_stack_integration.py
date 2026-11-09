"""Cube-stack deployment examples must compile without creating a simulator."""
import json
import os
from pathlib import Path
import subprocess
import sys

from pressb.experiments.config import PROJECT_ROOT, load_experiment
from pressb.registry import resolve


def test_cube_stack_registry_discovery_stays_lazy():
    code = """
from pressb.registry import resolve
import sys
assert resolve('task', 'cube_stack') == 'pressb.tasks.cube_stack.parallel:create_backend'
assert not any(key in sys.modules for key in ('torch', 'isaacsim'))
assert not any(key.startswith('pressb.tasks.') for key in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True,
                   env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")})


def test_cube_stack_node_and_experiment_share_task_contract():
    node = json.loads((PROJECT_ROOT / "configs/rl_nodes/cube_stack_simulation.json").read_text())
    compiled = load_experiment(PROJECT_ROOT / "configs/experiments/cube_stack_walloss_eval.json")
    simulation, inference, trainer = (compiled["nodes"][key] for key in ("simulation", "inference", "trainer"))
    assert compiled["factories"]["task"] == resolve("task", "cube_stack")
    assert node["backend"]["plugin"] == simulation["backend"]["plugin"] == "cube_stack"
    local = node["backend"]["config"]
    experiment = simulation["backend"]["config"]
    assert local["num_envs"] == experiment["num_envs"] == trainer["rollout"]["num_envs"] == 8
    assert local["horizon"] == experiment["horizon"] == trainer["rollout"]["execution_horizon"] == 32
    assert Path(experiment["scene_config"]) == PROJECT_ROOT / local["scene_config"]
    assert Path(experiment["scene_config"]).name == "playground_20261109_global_refined.json"
    for key in ("cubes", "randomization", "smoothing", "render_resolution", "reward", "terminate_on_success"):
        assert experiment[key] == local[key]
    assert "scene_overrides" not in experiment
    assert "reset_render_frames" not in experiment  # Preserve template warmup by default.
    assert node["server"] == simulation["server"]
    assert trainer["endpoints"]["simulation"] == "http://127.0.0.1:18423"
    assert trainer["endpoints"]["inference"] == "ws://127.0.0.1:18453"
    assert inference["backend"]["plugin"] == "walloss"
    assert inference["service"]["batch_size"] == 8


def test_real_layout_uses_same_template_and_cubes_with_sampling_disabled():
    base = json.loads((PROJECT_ROOT / "configs/rl_nodes/cube_stack_simulation.json").read_text())["backend"]["config"]
    real = json.loads((PROJECT_ROOT / "configs/rl_nodes/cube_stack_real_layout.json").read_text())["backend"]["config"]
    assert real["scene_config"] == base["scene_config"]
    assert real["cubes"] == base["cubes"]
    assert real["randomization"]["enabled"] is False
    assert base["randomization"]["enabled"] is True
    assert real["cubes"]["red"]["edge_m"] == 0.029
    assert real["cubes"]["blue"]["edge_m"] == 0.030
    assert "scene_overrides" not in real  # Keep the pinned home, not live feedback.


def test_generic_parallel_smoke_offers_cube_stack_without_starting_isaac():
    result = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts/smoke_parallel_tasks.py"), "--help"],
                            capture_output=True, text=True, check=True)
    assert "cube_stack" in result.stdout


def test_task_loaders_preserve_pinned_camera_and_home_in_both_layouts():
    from pressb.tasks.cube_stack.configuration import load_config, sample_objects
    reference = PROJECT_ROOT / "configs/calibration/playground_20261109_global_refined.json"
    original = reference.read_bytes()
    scene = json.loads(original)
    expected_initial = (scene["initial_state_joint_rad_m"] if "initial_state_joint_rad_m" in scene else
                        [*scene["initial_joint_positions_rad"], scene["initial_gripper_width_m"]])
    for name in ("cube_stack_simulation.json", "cube_stack_real_layout.json"):
        node = json.loads((PROJECT_ROOT / "configs/rl_nodes" / name).read_text())
        config = load_config(node["backend"]["config"])
        assert config["scene"]["initial_state_joint_rad_m"] == expected_initial
        assert config["scene"]["up_camera"] == scene["up_camera"]
        assert config["scene"]["wrist_camera"] == scene["wrist_camera"]
        assert config["reset_render_frames"] == scene["rendering"]["warmup_frames"]
        assert config["scene"]["task"] == "cube_stack"
        assert len(config["scene"]["objects"]) == 2
        objects = sample_objects(config, 0, 42)
        if not config["randomization"]["enabled"]:
            for color in ("red", "blue"):
                assert objects[color]["center_xy"] == config["cubes"][color]["center_xy"]
                assert objects[color]["yaw_deg"] == config["cubes"][color]["yaw_deg"]
    assert reference.read_bytes() == original
