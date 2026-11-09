"""CPU-only smoke checks using the real shared scheduler and schema-v2 service."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pressb.control.pose_protocol import pose8_to_pose9
from pressb.runtime.contract import encode_image
from pressb.runtime.parallel import SharedWorldBackend
from pressb.runtime.parallel_contract import ParallelTask

STATE = [0.3, 0., 0.4, 1., 0., 0., 0., .008]

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_elevator_task_backend.py"
spec = importlib.util.spec_from_file_location("elevator_task_smoke", SCRIPT)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class FixtureRobot:
    def __init__(self, engine):
        self.engine = engine
        self.q = np.zeros((engine.num_envs, 8))
        self.commands = np.zeros((engine.num_envs, 9))

    def cache_initial_state(self):
        pass

    def reset(self, ids, specs):
        self.q[ids] = 0.
        self.commands[ids] = 0.

    def begin_action(self, ids, actions):
        self.commands[ids] = actions
        return {i: self.commands[i].tolist() for i in ids}

    def before_tick(self, ids, substep, ticks_per_action):
        pass

    def observations(self, ids):
        return {i: dict(state=STATE, control_state=[0.]*25) for i in ids}

    def diagnostic_state(self, ids):
        return {i: dict(q=self.q[i].tolist(), commands=self.commands[i].tolist()) for i in ids}

    def close(self):
        pass


class TimedPhysics:
    def __init__(self, task, backend):
        self.task, self.backend = task, backend
        self.num_envs = backend.num_envs
        self.world = SimpleNamespace(current_time=0., current_time_step_index=0)
        self.robot = FixtureRobot(self)

    def initialize(self):
        self.task.bind(self)

    def check(self):
        pass

    def tick(self, ids):
        self.world.current_time_step_index += 1
        self.world.current_time = self.world.current_time_step_index / 120

    def publish_transforms(self):
        pass

    def reset_render_history(self):
        pass

    def capture(self, ids, render_frames, reset_history=True):
        image = encode_image(np.zeros((8,8,3),np.uint8))
        return {i: {"global": image, "wrist": image} for i in ids}

    def close(self):
        pass


class FixtureTask(ParallelTask):
    def __init__(self, config, leak=False):
        self.config, self.leak = config, leak
        self.execution = dict(num_envs=config["num_envs"], gpu=0, physics_dt=1/120,
            control_hz=30., horizon=7, max_episode_seconds=config["max_seconds"], gamma=.99,
            terminal_boundary="physics_tick", timeout_reason="time_limit")
        self.generations = [0]*config["num_envs"]

    def metadata(self):
        return dict(schema_version=2, node_type="simulation", task_id="elevator", num_envs=2,
            observation=dict(state=dict(names=["x","y","z","qw","qx","qy","qz","gripper"],
                units=["m"]*3+["unitless"]*4+["m"]), cameras=["global","wrist"]),
            action=dict(representation="pose9",frame="robot_base",names=[f"p{i}" for i in range(9)],
                units=["m"]*3+["unitless"]*6,horizon=7,control_hz=30.),
            control_state_dim=25,task_features_dim=0,discount=dict(gamma=.99,timebase_hz=30.),
            reset_mode="indexed")

    def sample_reset(self, eid, seed, options):
        floor = options.get("floor",24)
        return {**options,"floor":floor,"task":f"Press {floor} floor."}

    def reset_task(self, ids, specs):
        for i in ids:
            self.generations[i] += 1
        if self.leak and len(ids) == 1:
            self.engine.robot.q[1-ids[0],0] += 1
        return {i:{} for i in ids}

    def after_tick(self, ids):
        rows = {}
        for i in ids:
            success = bool(self.generations[i]>1 and self.engine.backend.reset_specs[i]["floor"]==33
                           and self.engine.backend.ticks[i]>=148)
            reason = "target_pressed" if success else "time_limit" if self.engine.backend.ticks[i]>=round(
                self.config["max_seconds"]*120) else None
            rows[i] = dict(success=success,terminated=success,reward=float(success),reason=reason,
                info={"termination":reason})
        return rows


class FakeNative(SharedWorldBackend):
    def __init__(self, config, *, leak=False):
        self.was_closed = False
        super().__init__(FixtureTask(config,leak=leak),engine_factory=TimedPhysics)

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
    assert (args.gpu_id, args.num_envs, args.max_seconds, args.gpu_dynamics) == (1, 2, 2., False)


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
    assert "Indexed reset changed a peer" in report["error"]


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
