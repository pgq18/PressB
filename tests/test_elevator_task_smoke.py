"""CPU-only smoke checks using real schema-v2 and legacy service state machines."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pressb.online_rl.fast_simulation import IndexedSimulationService
from pressb.online_rl.protocol import pose8_to_pose9
from pressb.tasks.elevator.backend import ElevatorBackend
from test_rl_nodes_elevator import LocalClient, PhysicsFixture, STATE

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_elevator_task_backend.py"
spec = importlib.util.spec_from_file_location("elevator_task_smoke", SCRIPT)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class TimedPhysics(PhysicsFixture):
    def __init__(self, config, *, leak=False):
        super().__init__()
        self.config = config
        self.leak = leak
        self.world = SimpleNamespace(current_time=0., current_time_step_index=0)
        self.q = np.zeros((2, 8))
        self.velocity = self.q.copy()
        self._commands = self.q.copy()
        self.button_rows = np.arange(24).reshape(2, 12)
        self.positions = np.zeros((24, 3))
        self.orientations = np.tile([1., 0., 0., 0.], (24, 1))
        self.button_velocity = np.zeros((24, 6))
        self.button_view = SimpleNamespace(
            get_world_poses=lambda: (self.positions.copy(), self.orientations.copy()),
            get_velocities=lambda: self.button_velocity.copy())
        self._items = {i: {"ticks": 0, "done": False} for i in range(2)}
        self.generations = [0, 0]
        self.timing = {"chunks": 0}

    def health(self):
        return {**super().health(), "max_seconds": self.config["max_seconds"]}

    def _joint_arrays(self):
        return self.q.copy(), self.velocity.copy()

    def _reset_slots(self, ids):
        for i in ids:
            self._items[i] = {"ticks": 0, "done": False}
            self.generations[i] += 1
            self.q[i] = self.generations[i]
        if self.leak and len(ids) == 1:
            self.q[1 - ids[0], 0] += 1

    def reset(self, episodes, seed):
        self._reset_slots([0, 1])
        return super().reset(episodes, seed)

    def reset_envs(self, episodes, seed):
        self._reset_slots(list(episodes))
        return super().reset_envs(episodes, seed)

    def step(self, actions):
        self.calls.append(("step", actions))
        rows, advances = [], []
        for i in actions:
            success_episode = self.generations[i] > 1 and self.episodes[i]["floor"] == 33
            limit = 148 if success_episode else round(self.config["max_seconds"] * 120)
            ticks = min(28, limit - self._items[i]["ticks"])
            self._items[i]["ticks"] += ticks
            done = self._items[i]["ticks"] == limit
            self._items[i]["done"] = done
            row = self.observe(i)
            row.update(terminated=bool(done and success_episode), truncated=bool(done and not success_episode),
                executed_physics_steps=ticks, info={"success": bool(done and success_episode),
                    "termination": "target_pressed" if done and success_episode else "time_limit" if done else None})
            rows.append(row)
            advances.append(ticks)
        self.world.current_time_step_index += max(advances)
        self.world.current_time = self.world.current_time_step_index / 120
        self.timing["chunks"] += 1
        return rows


class FakeNative(ElevatorBackend):
    def __init__(self, config, *, leak=False):
        self._engine = TimedPhysics(config, leak=leak)
        client = LocalClient()
        client.physics = self._engine
        client.service = IndexedSimulationService(self._engine, single_gamma=.99)
        self._legacy_service = client.service
        super().__init__({"endpoint": "http://unused", "floors": [24, 33]}, client=client)
        self.was_closed = False

    def close(self):
        self.was_closed = True
        super().close()


@pytest.fixture
def replay_log(tmp_path):
    path = tmp_path / "policy_actions.jsonl"
    chunk = pose8_to_pose9(np.tile(STATE, (7, 1))).tolist()
    rows = [{"episode_id": "success", "physics_index": i * 28,
        "combined_actions_pose9": chunk} for i in range(6)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "episodes.jsonl").write_text(json.dumps({"episode_id": "success", "success": True,
        "sim_seconds": 148 / 120, "layout": {"floor": 33, "offset_x_m": 0., "offset_y_m": 0.}}) + "\n")
    return path


def arguments(tmp_path, replay_log):
    return smoke.parser().parse_args(["--output-dir", str(tmp_path / "run"),
        "--replay-log", str(replay_log), "--replay-episode-id", "success"])


def test_help_uses_no_backend_or_simulator_imports():
    code = """
import importlib.abc
import runpy
import sys
class RejectImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("pressb", "isaacsim", "omni", "carb", "pxr"):
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectImports())
sys.argv = [sys.argv[1], "--help"]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
    result = subprocess.run([sys.executable, "-c", code, str(SCRIPT)],
        capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    assert "--replay-log" in result.stdout
    args = smoke.parser().parse_args([])
    assert (args.gpu_id, args.num_envs, args.max_seconds, args.gpu_dynamics) == (1, 2, 2., True)


def test_replay_loader_keeps_complete_chunks_and_source_layout(replay_log):
    chunks, layout, evidence = smoke.load_replay(replay_log, "success")
    assert np.asarray(chunks).shape == (6, 7, 9)
    assert layout == {"floor": 33, "offset_x_m": 0., "offset_y_m": 0.}
    assert evidence["source_sim_seconds"] == 148 / 120
    assert len(evidence["log_sha256"]) == 64
    with pytest.raises(RuntimeError, match="not found"):
        smoke.load_replay(replay_log, "missing")


def test_full_smoke_wraps_real_services_and_replays_both_slots(tmp_path, replay_log):
    args = arguments(tmp_path, replay_log)
    instances = []

    def factory(config):
        instances.append(FakeNative(config))
        return instances[-1]

    assert smoke.run(args, backend_factory=factory) == 0

    assert len(instances) == 1 and instances[0].was_closed
    report = json.loads((args.output_dir / "report.json").read_text())
    assert report["status"] == "passed" and report["closed"]
    assert report["stages"][0]["physics_steps_by_chunk"][-1] == {"0": 16, "1": 16}
    assert all(row["physics_steps"] == 240 for row in report["stages"][0]["episodes"])
    assert all(row["success"] for row in report["stages"][1]["episodes"])
    assert report["indexed_reset"][0]["peer_env_ids"] == [1]
    assert report["indexed_reset"][1]["peer_env_ids"] == [0]
    assert all(check["physical_state_unchanged"] for check in report["indexed_reset"])
    assert len(list(args.output_dir.glob("*/*.png"))) == 16
    assert report["close_run"] == {"closed": True, "restart_required": False}


def test_peer_mutation_fails_smoke_and_still_closes(tmp_path, replay_log):
    args = arguments(tmp_path, replay_log)
    instances = []

    def factory(config):
        instances.append(FakeNative(config, leak=True))
        return instances[-1]

    assert smoke.run(args, backend_factory=factory) == 1
    assert instances[0].was_closed
    report = json.loads((args.output_dir / "report.json").read_text())
    assert "peer physical buffers" in report["error"]


def test_report_survives_fast_shutdown_before_close_returns(tmp_path, replay_log):
    args = arguments(tmp_path, replay_log)

    class ExitingNative(FakeNative):
        def close(self):
            report = json.loads((args.output_dir / "report.json").read_text())
            assert report["operations_status"] == "passed"
            assert report["status"] == "closing" and not report["closed"]
            raise SystemExit(0)

    with pytest.raises(SystemExit) as error:
        smoke.run(args, backend_factory=ExitingNative)
    assert error.value.code == 0


@pytest.mark.parametrize("options", [["--num-envs", "1"], ["--max-seconds", "0"], ["--max-seconds", "0.001"]])
def test_invalid_cli_fails_before_backend_construction(options):
    with pytest.raises(SystemExit) as error:
        smoke.main(options)
    assert error.value.code == 2
