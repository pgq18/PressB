"""No Isaac import: recorder must preserve state, control timing and terminal state."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pressb.tasks.pap.recording import PAPStateRecorder, PAPRecordingVectorBackend


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
        self.execution = self.config
        self.engine = SimpleNamespace(offsets=self.offsets, robot=SimpleNamespace(arms=self.arms, commands=self.commands))
        self.task = SimpleNamespace(identity=self.identity, contact_counts=self.contact_counts)

    def describe(self):
        return {"task_id": "pap"}

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
        world_tick = tick
        backend.tick = tick
        backend.ticks[backend.ready] += 1
        if tick == 3:
            backend.ready[0] = False
        backend.recorder.after_tick(backend, [i for i in range(2) if backend.ready[i] or (i == 0 and tick == 3)],
            {i: {"success": i == 0 and tick == 3, "reason": "success" if i == 0 and tick == 3 else None} for i in range(2)})
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
    from pressb.tasks.pap.backend import ROOT, task_identity
    from pressb.tasks.pap.vector import load_config
    raw = json.loads((ROOT/"configs/rl_nodes/pap_simulation.json").read_text())["backend"]["config"]
    recorded = load_config({**raw, "num_envs": 64, "gpu_ids": [0]})
    reconstruct = replay_loader().__globals__["replay_config"]
    replay = load_config(reconstruct({"config": recorded}, tmp_path, 1))
    assert replay["objects"] == recorded["objects"]
    assert replay["scene"] == recorded["scene"]
    assert task_identity(replay)["task_config_sha256"] == task_identity(recorded)["task_config_sha256"]
    assert replay["num_envs"] == 1
    assert replay["gpu"] == 1


@pytest.mark.parametrize("fused", [False, True])
def test_shared_constructor_records_exact_terminal_and_no_extra_work(tmp_path, monkeypatch, fused):
    """Exercise the real PAP constructor and runtime, replacing only Isaac I/O."""
    import sys
    from pressb.runtime.parallel import SharedWorldBackend
    from pressb.tasks.pap.backend import ROOT
    from pressb.tasks.pap.parallel import PAPTask
    from pressb.tasks.pap.vector import PAPVectorBackend

    shader = lambda path: SimpleNamespace(GetInput=lambda name: SimpleNamespace(Set=lambda value: None))
    monkeypatch.setitem(sys.modules, "pxr", SimpleNamespace(Gf=SimpleNamespace(Vec3f=lambda *v: tuple(v)),
                                                         UsdShade=SimpleNamespace(Shader=shader)))

    class FakeBodies:
        def __init__(self, z):
            self.p = np.array([[.3, 0., z], [6.3, 0., z]])
            self.q = np.tile([1., 0., 0., 0.], (2, 1))
            self.v = np.zeros((2, 6))
        def get_world_poses(self): return self.p.copy(), self.q.copy()
        def get_velocities(self): return self.v.copy()
        def set_world_poses(self, p, q, ids): self.p[ids], self.q[ids] = p, q
        def set_velocities(self, v, ids): self.v[ids] = v

    class FakeRobot:
        def __init__(self, task):
            self.state = np.tile(task.initial, (2, 1))
            self.initial = self.state.copy()
            self.commands = np.c_[self.state[:, :6], self.state[:, 6]/2, -self.state[:, 6]/2]
            self.arms = SimpleNamespace(view=SimpleNamespace(shared_metatype=SimpleNamespace(
                dof_names=[f"joint{i}" for i in range(1, 9)])))
            self.postprocessors = [SimpleNamespace(summary=lambda: {"mode": "none"}) for _ in range(2)]
        def reset(self, ids, specs): self.state[ids] = self.initial[ids]
        def observations(self, ids):
            return {i: {"state": self.state[i].tolist(), "control_state": self.state[i].tolist()} for i in ids}
        def diagnostic_state(self, ids):
            return {i: {"state": self.state[i].tolist(), "joints": self.commands[i].tolist(),
                        "joint_velocity": [0.]*8} for i in ids}
        def begin_action(self, ids, values):
            self.state[ids] = values
            return {i: self.state[i] for i in ids}
        def before_tick(self, *args): pass

    class FakeEngine:
        def __init__(self, task, backend):
            self.task, self.backend = task, backend
            self.num_envs = backend.num_envs
            self.offsets = np.array([[0., 0., 0.], [6., 0., 0.]])
            self.world = SimpleNamespace(current_time=0., current_time_step_index=0)
            self.stage = SimpleNamespace(GetPrimAtPath=lambda path: path)
            self.robot = FakeRobot(task)
            self.bodies = FakeBodies(task.config["scene"]["table_height"]+.015)
            self.template = SimpleNamespace(robot_path="/World/Piper", data={
                "scene": {"box_inside": {"origin_world": [10., 0., task.config["scene"]["table_height"]],
                    "yaw_deg": 0., "min_local": [-.05, -.075, 0.], "max_local": [.05, .075, .06]}},
                "base": np.zeros(3), "object_paths": ["/World/Object"], "material_paths": ["/World/Material"]})
            self.closed, self.captures = False, 0
        def initialize(self):
            self.task.bind(self)
            self.task.after_settle()
        def relocate(self, eid, path): return f"/World/envs/env_{eid}"+path[6:]
        def rigid_bodies(self, paths): return self.bodies, np.array([[0], [1]])
        def check(self): pass
        def publish_transforms(self): pass
        def reset_render_history(self): pass
        def tick(self, ids):
            self.world.current_time_step_index += 1
            self.world.current_time = self.world.current_time_step_index*self.task.config["physics_dt"]
            self.bodies.p[:, 0] += .001
            if self.backend.ticks[0] >= 3:
                self.bodies.p[0, 2] = self.task.config["scene"]["table_height"]-.1
        def capture(self, ids, render_frames, reset_history=True):
            self.captures += render_frames
            return {i: {name: {"encoding": "fake"} for name in ("up", "wrist")} for i in ids}
        def close(self): self.closed = True

    cfg = json.loads((ROOT/"configs/rl_nodes/pap_vector_simulation.json").read_text())["backend"]["config"]
    cfg.update(num_envs=2, gpu_ids=[0], max_episode_seconds=8/120, reset_render_frames=1,
               observation_render_frames=1, settling_ticks=1, smoothing={"mode": "none", "alpha": .8})
    recording_cfg = {**cfg, "trajectory_recording": {"directory": str(tmp_path/"recorded"),
        "env_ids": [0, 1], "max_episodes_per_env": 2}}
    recorded = PAPRecordingVectorBackend(recording_cfg, engine_factory=FakeEngine)
    plain = PAPVectorBackend(cfg, engine_factory=FakeEngine)
    assert isinstance(recorded, SharedWorldBackend) and isinstance(recorded.task, PAPTask)
    assert recorded.describe()["trajectory_recording"]["additional_render_or_physics_calls"] == 0
    try:
        for backend in (recorded, plain):
            backend.reset([0, 1], 4, {"color": "red"})
            commands = [{"env_id": i, "actions": np.tile(backend.task.initial, (2, 1)).tolist()} for i in range(2)]
            if fused:
                backend.step_and_reset(commands, reset_seed=5, reset_options={"color": "blue"}, max_active_envs=2)
            else:
                backend.step(commands, terminal_observation_mode="state_only_if_no_bootstrap")
        assert recorded.engine.world.current_time_step_index == plain.engine.world.current_time_step_index == 8
        assert recorded.engine.captures == plain.engine.captures
        assert recorded.snapshot() == plain.snapshot()
        for eid, ticks in ((0, [0, 3]), (1, [0, 4, 8])):
            path = tmp_path/"recorded"/f"env_{eid:03d}_episode_000"
            meta = json.loads(path.with_suffix(".json").read_text())
            assert meta["complete"] and meta["final_physics_tick"] == ticks[-1]
            with np.load(path.with_suffix(".npz")) as data:
                assert data["episode_physics_steps"].tolist() == ticks
                assert data["world_time"].tolist() == pytest.approx(np.array(ticks)/120)
                assert np.diff(data["object_position_world"][:, 0]).tolist() == pytest.approx(np.diff(ticks)*.001)
        if fused:
            assert recorded.recorder.slots[0]["samples"][0]["episode_physics_steps"] == 0
            assert recorded.recorder.slots[0]["reset"]["color"] == "blue"
    finally:
        recorded.close()
        plain.close()
    assert recorded.engine.closed and not recorded.recorder.slots
