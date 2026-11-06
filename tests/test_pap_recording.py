"""No Isaac import: recorder must preserve state, control timing and terminal state."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pressb.rl_nodes.pap_recording import PAPStateRecorder, PAPRecordingVectorBackend


class Backend:
    def __init__(self):
        self.num_envs = 2
        self.config = {"physics_dt": 1/120, "control_hz": 30, "physics_ticks_per_action": 4}
        self.identity = {"task_config_sha256": "dummy"}
        self.arms = SimpleNamespace(view=SimpleNamespace(shared_metatype=SimpleNamespace(dof_names=["joint1"])))
        self.offsets = np.array([[0., 0., 0.], [0., 6., 0.]])
        self.commands = np.zeros((2, 1))
        self.contact_counts = np.zeros((2, 2), dtype=int)
        self.reset_specs = [{"color": "red"}, {"color": "blue"}]
        self.tick = 0

    def _ids(self, ids):
        if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids) or any(i not in [0, 1] for i in ids):
            raise ValueError("invalid ids")

    def snapshot(self, ids):
        return {str(i): dict(state=[float(self.tick)], joints=[float(self.tick)], joint_velocity=[0.],
            object_position_world=[float(self.tick), 0., .1], object_quaternion_wxyz=[1., 0., 0., 0.],
            object_velocity=[0.]*6, episode_physics_steps=self.tick, ready=True,
            world_time=2+self.tick/120) for i in ids}


def replay_loader():
    path = Path(__file__).resolve().parents[1]/"scripts/render_pap_recorded_rollout.py"
    spec = importlib.util.spec_from_file_location("pap_replay_test_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_episode


def test_measured_terminal_tick_and_read_only(tmp_path):
    backend = Backend()
    recorder = PAPStateRecorder(backend, {"directory": str(tmp_path), "env_ids": [1]})
    before = copy.deepcopy(backend.snapshot([0, 1]))
    recorder.begin([0, 1])
    assert backend.snapshot([0, 1]) == before
    backend.tick = 4
    recorder.append([0, 1])
    backend.tick = 7  # termination before the next 30 Hz boundary
    recorder.append([1], {1: {"success": True}}, {1: "success"})
    recorder.finish(1, complete=True, info={"episode_physics_steps": 7, "success": True, "termination_reason": "success"})
    meta, manifest, data, indices = replay_loader()(tmp_path/"env_001_episode_000.json")
    assert data["episode_physics_steps"].tolist() == [0, 4, 7]
    assert data["joints"].ravel().tolist() == [0, 4, 7]
    assert data["success"].tolist() == [False, False, True]
    assert indices.tolist() == [0, 1, 2]
    assert meta["duration_seconds"] == 7/120
    assert not list(tmp_path.glob("env_000*"))
    assert backend.tick == 7


def test_reset_discards_no_record_silently_and_respects_limit(tmp_path):
    backend = Backend()
    recorder = PAPStateRecorder(backend, {"directory": str(tmp_path), "env_ids": [0]})
    recorder.begin([0])
    recorder.begin([0])
    recorder.close()
    rows = [json.loads(line) for line in (tmp_path/"index.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and not rows[0]["complete"]
    assert recorder.slots == {}


def test_partial_record_is_not_presented_as_complete(tmp_path):
    recorder = PAPStateRecorder(Backend(), {"directory": str(tmp_path), "env_ids": [0]})
    recorder.begin([0])
    recorder.close()
    with pytest.raises(ValueError, match="complete PAP"):
        replay_loader()(tmp_path/"env_000_episode_000.json")


def test_refuses_existing_recording(tmp_path):
    (tmp_path/"do_not_replace").write_text("existing")
    with pytest.raises(ValueError, match="empty"):
        PAPStateRecorder(Backend(), {"directory": str(tmp_path)})


def test_wrapper_records_terminal_before_peer_physics_advances(tmp_path):
    # A terminal object keeps moving while peers continue; the recorded terminal
    # pose must be the earlier state, even though finish() occurs after the batch.
    backend = Backend()
    backend.ticks = np.zeros(2, dtype=int)
    backend.ready = np.ones(2, dtype=bool)
    backend.recorder = PAPStateRecorder(backend, {"directory": str(tmp_path), "env_ids": [0, 1]})
    world_tick = 0

    def snapshot(ids):
        result = Backend.snapshot(backend, ids)
        for i in ids:
            result[str(i)]["episode_physics_steps"] = int(backend.ticks[i])
            result[str(i)]["object_position_world"][0] = float(world_tick)
            result[str(i)]["world_time"] = 2+world_tick/120
        return result

    backend.snapshot = snapshot
    backend.recorder.begin([0, 1])
    for tick in range(1, 9):
        PAPRecordingVectorBackend._record_due_states(backend)  # pre-world-step
        world_tick = tick
        backend.tick = tick
        backend.ticks[backend.ready] += 1
        if tick == 3:
            backend.ready[0] = False
    PAPRecordingVectorBackend._record_due_states(backend)  # final _step boundary
    backend.recorder.finish(0, complete=True, info={"episode_physics_steps": 3, "success": True,
        "termination_reason": "success"})
    with np.load(tmp_path/"env_000_episode_000.npz") as data:
        assert data["episode_physics_steps"].tolist() == [0, 3]
        assert data["object_position_world"][:, 0].tolist() == [0., 3.]
    assert [row["episode_physics_steps"] for row in backend.recorder.slots[1]["samples"]] == [0, 4, 8]


def test_physx_float32_clock_is_validated_without_changing_playback_dt():
    audit = replay_loader().__globals__["audit_clock"]
    dt = 1/120
    ticks = np.array([0, 4, 734])
    data = {"episode_physics_steps": ticks, "world_time": 2+ticks*float(np.float32(dt))}
    result = audit(data, dt)
    assert result["max_world_vs_runtime_error_seconds"] == 0
    assert 3e-7 < result["max_world_vs_declared_error_seconds"] < 3.3e-7
    assert result["declared_physics_dt_seconds"] == dt
    data["world_time"][-1] += 1e-6  # small but unrelated clock error must fail
    with pytest.raises(ValueError, match="float32 dt"):
        audit(data, dt)


def test_replay_reconstructs_raw_objects_without_nested_geometry(tmp_path):
    from pressb.rl_nodes.pap import ROOT, task_identity
    from pressb.rl_nodes.pap_vector import load_config
    raw = json.loads((ROOT/"configs/rl_nodes/pap_simulation.json").read_text())["backend"]["config"]
    recorded = load_config({**raw, "num_envs": 64, "gpu_ids": [0]})
    reconstruct = replay_loader().__globals__["replay_config"]
    replay = load_config(reconstruct({"config": recorded}, tmp_path, 1))
    assert replay["objects"] == recorded["objects"]
    assert replay["scene"] == recorded["scene"]
    assert task_identity(replay)["task_config_sha256"] == task_identity(recorded)["task_config_sha256"]
    assert replay["num_envs"] == 1
    assert replay["gpu"] == 1
