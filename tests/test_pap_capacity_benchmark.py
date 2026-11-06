"""Capacity-probe correctness without importing Isaac or allocating any GPU."""
import base64
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pap_capacity_benchmark", ROOT / "scripts/benchmark_pap_parallel.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_watchdog_distinguishes_memory_pressure_and_monitor_loss():
    sample = {"host_available_bytes": 16*benchmark.GIB,
              "gpu": {"devices": [{"index": 0, "free_mib": 6000}]}}
    assert benchmark.resource_reason(sample, [0], 8, 2) is None
    assert benchmark.resource_reason(sample, [1], 8, 2) == "gpu_monitor_unavailable"
    assert benchmark.resource_reason(sample, [0], 20, 2) == "host_available_below_guard"
    assert benchmark.resource_reason(sample, [0], 8, 8) == "gpu_free_below_guard"


def test_json_payload_sizes_include_protocol_envelope():
    request, response = {"items": []}, {"items": [], "label": "PAP"}
    result = benchmark.json_payload_evidence(request, response)
    assert result["request_json_bytes"] == len(json.dumps({**request, "protocol_version": 1}).encode())
    assert result["response_json_bytes"] == len(json.dumps({**response, "protocol_version": 1},
        separators=(",", ":")).encode())
    assert result["within_64mib_budget"]


class FakePAP:
    """Tracks true per-environment state, including a deliberate reset-leak mode."""
    def __init__(self, leak=False):
        self.q = np.zeros((3, 7))
        self.ticks = [0]*3
        self.closed = False
        self.leak = leak
        array = np.zeros((32, 32, 3), dtype=np.uint8)
        array[:, :16] = [220, 10, 30]
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="PNG")
        self.image = {"encoding": "png_base64", "data": base64.b64encode(buffer.getvalue()).decode()}

    def describe(self):
        vector = {"names": [f"joint{i}" for i in range(7)], "units": ["rad"]*6+["m"]}
        return dict(schema_version=2, node_type="simulation", task_id="synthetic-capacity-fixture",
            num_envs=3, observation={"state": vector, "cameras": ["up", "wrist"]},
            action={**vector, "horizon": 32, "control_hz": 30, "representation": "joint_position",
                    "frame": "robot_base", "lower": [-1]*7, "upper": [1]*7},
            control_state_dim=7, task_features_dim=0, discount={"gamma": .99, "timebase_hz": 30},
            reset_mode="indexed", scheduling="all_active")

    def observation(self, i):
        return dict(state=self.q[i].tolist(), control_state=self.q[i].tolist(),
                    task_features=[], task="Synthetic fixture.", images={"up": self.image, "wrist": self.image})

    def reset(self, env_ids, seed, options):
        for i in env_ids:
            self.q[i] = 0
            self.ticks[i] = 0
        if self.leak:
            self.ticks = [0]*3
        return [dict(env_id=i, observation=self.observation(i),
                     info={"reset": {"center_xy": [.1+i*.01, .2]}}) for i in env_ids]

    def step(self, commands):
        rows = []
        for command in commands:
            i = command["env_id"]
            self.q[i] = command["actions"][-1]
            self.ticks[i] += len(command["actions"])
            rows.append(dict(env_id=i, observation=self.observation(i), reward=0.,
                discount=.99**len(command["actions"]), terminated=False, truncated=self.ticks[i] >= 60,
                info={"success": False, "physics_steps": 4*len(command["actions"]),
                      "elapsed_control_steps": len(command["actions"]),
                      "episode_physics_steps": self.ticks[i]*4,
                      "termination_reason": "timeout" if self.ticks[i] >= 60 else None}))
        return rows

    def diagnostic_snapshot(self, ids):
        return [{"ticks": self.ticks[i], "q": self.q[i].tolist()} for i in ids]

    def close(self):
        self.closed = True


@pytest.mark.parametrize("leak", [False, True])
def test_child_measures_chunk_transitions_and_detects_reset_leak(tmp_path, monkeypatch, leak):
    from pressb.rl_nodes import server
    plant = FakePAP(leak=leak)
    monkeypatch.setattr(server, "load_backend", lambda config: plant)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"backend": {}}))
    args = benchmark.parser().parse_args(["--child", "--config", str(config),
        "--output-dir", str(tmp_path), "--chunks", "4"])
    code = benchmark.run_child(args)
    report = json.loads((tmp_path / "simulation_results.json").read_text())
    assert plant.closed
    if leak:
        assert code == 1 and "physical diagnostic state" in report["error"]
        return
    assert code == 0 and report["state"] == "completed"
    assert report["transitions"] == 12
    assert report["control_steps"] == 384
    assert len(report["episode_endings"]) == 6
    assert report["unique_initial_object_centers"] == 3
    assert all(row["physical_snapshot_unchanged"] for row in report["indexed_reset_checks"])
    assert len(set(report["first_chunk_joint1"])) == 3
    assert report["success_rate_not_evaluated"]
    assert report["json_payloads"]["all_within_64mib_budget"]
    assert sum(row["method"] == "step" for row in report["json_payloads"]["calls"]) == 4
    for camera in ("up", "wrist"):
        assert len(report["images"][camera]["images"]) == 3
        assert Path(report["images"][camera]["contact_sheet"]).is_file()


def test_resolve_config_preserves_original_and_removes_stale_worker_count(tmp_path):
    source = {"backend": {"factory": "old:factory", "config": {
        "workers": [{"gpu": 0}], "custom_task_parameter": 42}}}
    path = tmp_path / "source.json"
    path.write_text(json.dumps(source))
    args = benchmark.parser().parse_args(["--config", str(path), "--output-dir", str(tmp_path),
        "--factory", "new:factory", "--num-envs", "64", "--gpu-id", "1",
        "--reset-render-frames", "8", "--render-subframes", "2"])
    result = benchmark.resolved_config(args)
    assert json.loads(path.read_text()) == source
    assert result["backend"]["factory"] == "new:factory"
    assert result["backend"]["config"]["num_envs"] == 64
    assert result["backend"]["config"]["gpu_ids"] == [1]
    assert "workers" not in result["backend"]["config"]
    assert result["backend"]["config"]["custom_task_parameter"] == 42
    assert result["backend"]["config"]["reset_render_frames"] == 8
    assert result["backend"]["config"]["render_subframes"] == 2


def replay_fixture(tmp_path):
    episode = tmp_path / "episode"
    episode.mkdir()
    scene = {"fps": 30, "physics_dt": 1/120, "cube": {"center_xy": [.3, .02], "yaw_deg": 0}}
    (episode / "config.json").write_text(json.dumps(scene))
    commands = np.zeros((64, 8))
    commands[:, 0] = np.linspace(0, .02, 64)
    commands[:, 6], commands[:, 7] = .01, -.01
    target = np.r_[np.zeros((1, 8)), np.repeat(commands, 4, axis=0)]
    np.savez(episode / "physics.npz", q_target=target)
    return episode, target


def test_replay_requires_constant_applied_command_per_control_interval(tmp_path):
    episode, target = replay_fixture(tmp_path)
    actions, options, metadata = benchmark.load_replay(episode, 30)
    assert actions.shape == (64, 7)
    assert np.allclose(actions[:, 6], .02)
    assert options["center_xy"] == [.3, .02]
    assert metadata["already_smoothed"]
    target[3, 0] += .001
    np.savez(episode / "physics.npz", q_target=target)
    with pytest.raises(ValueError, match="within a control interval"):
        benchmark.load_replay(episode, 30)


def test_replay_records_physical_outcomes_without_policy_success_claim(tmp_path, monkeypatch):
    from pressb.rl_nodes import server
    episode, _ = replay_fixture(tmp_path)
    plant = FakePAP()
    monkeypatch.setattr(server, "load_backend", lambda config: plant)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"backend": {"config": {"max_episode_seconds": 20}}}))
    args = benchmark.parser().parse_args(["--child", "--config", str(config),
        "--output-dir", str(tmp_path), "--replay-episode", str(episode)])
    assert benchmark.run_child(args) == 0
    report = json.loads((tmp_path / "simulation_results.json").read_text())
    assert report["mode"] == "recorded_action_physics_qualification"
    assert report["transitions"] == 6
    assert report["replay_success_count"] == 0
    assert all(row["truncated"] for row in report["replay_outcomes"])
    assert report["policy_success_rate_not_evaluated"]
    assert plant.closed
    assert (tmp_path / "final_up_all_envs.jpg").is_file()


def test_replay_configuration_disables_double_smoothing_and_uses_source_scene(tmp_path):
    episode, _ = replay_fixture(tmp_path)
    path = tmp_path / "source.json"
    path.write_text(json.dumps({"backend": {"factory": "a:b", "config": {}}}))
    args = benchmark.parser().parse_args(["--config", str(path), "--output-dir", str(tmp_path),
        "--replay-episode", str(episode)])
    config = benchmark.resolved_config(args)["backend"]["config"]
    assert config["scene_config"] == str(episode / "config.json")
    assert config["max_episode_seconds"] == 20
    assert config["smoothing"]["mode"] == "none"
    assert config["randomization"]["position"] == "fixed"
