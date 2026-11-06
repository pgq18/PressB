"""CPU checks for PAP sampling, parallel dispatch, and physical-time semantics.

These do not assert Isaac rendering/PhysX integration. A real smoke run must be
performed separately when a GPU has sufficient memory for another application.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from pressb.action_postprocessing import JointActionPostprocessor
from pressb.rl_nodes.pap import PAPParallelBackend, load_config, sample_reset, task_identity
from pressb.rl_nodes.pap_worker import PAPEnvironment
from pressb.tasks.pap.task import PickPlaceOutcome
from pressb.rl_nodes import pap


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "episode_000000.npz"
    np.savez(path, state_raw=np.zeros((1, 7)))
    return load_config({"python": sys.executable, "trajectory": str(path),
        "num_envs": 2, "gpu_ids": [0, 1], "max_episode_seconds": .1,
        "randomization": {"position": "markers", "colors": ["red", "blue", "yellow"]}})


def test_randomization_uses_marker_edges_and_is_independent(config):
    first = sample_reset(config, 2345, 0, {})
    assert first == sample_reset(config, 2345, 0, {})
    assert first["center_xy"] != sample_reset(config, 2345, 1, {})["center_xy"]
    assert first["sampling_region"]["region_kind"] == "rectangle_from_marker_long_axis_lines"
    planes = np.asarray(first["sampling_region"]["valid_center_halfplanes"])
    for seed in range(100):
        position = sample_reset(config, seed, 0, {})["center_xy"]
        assert np.max(planes[:, :2] @ position + planes[:, 2]) < 1e-10


def test_randomization_changes_object_geometry_and_preserves_overrides(config):
    cfg = copy.deepcopy(config)
    cfg["objects"].append({"shape": "cylinder", "radius": .018, "height": .025,
        "geometry": {"shape": "cylinder", "radius": .018, "height": .025, "size_xyz": [.036, .036, .025]},
        "yaw_deg": 0., "color": [1., 0., 0.], "center_xy": [.3, 0.]})
    first = sample_reset(cfg, 1, 0, {"object_index": 1, "color": "blue", "center_xy": [.32, -.03],
        "task": "Put the blue object in the box.", "model_inputs": {"dataset_name": "custom"}})
    assert first["object_index"] == 1
    assert first["center_xy"] == [.32, -.03]
    assert first["color_rgb"] == [.006, .36, .85]
    assert first["model_inputs"] == {"dataset_name": "custom"}
    assert first["sampling_region"]["edge_m"] == .036


def test_identity_tracks_physical_task_but_not_worker_placement(config):
    original = task_identity(config)["task_config_sha256"]
    moved = copy.deepcopy(config)
    moved.update(num_envs=9, workers=[{"gpu": 0}]*9, startup_timeout=900,
        log_dir="/different/location", command_timeout=1.)
    assert task_identity(moved)["task_config_sha256"] == original
    for key, value in (("gamma", .995), ("reward", {"success": 2., "failure": 0.}),
                       ("smoothing", {"mode": "joint_ema", "alpha": .8})):
        changed = {**config, key: value}
        assert task_identity(changed)["task_config_sha256"] != original


@pytest.mark.parametrize("update", [
    {"gamma": 1.01}, {"num_envs": True}, {"workers": [{"gpu": 0}]},
    {"smoothing": {"mode": "joint_ema", "alpha": 0}},
    {"randomization": {"colors": ["purple"]}}, {"randomization": {"position": "vertices"}},
    {"scene_overrides": {"task_props": False}},
    {"workers": [{"gpu": 0, "gamma": .5}, {"gpu": 0}]},
])
def test_configuration_errors_rejected_before_worker_start(config, update):
    raw = {"python": sys.executable, "trajectory": config["trajectory"], "num_envs": 2, **update}
    with pytest.raises(ValueError):
        load_config(raw)


class FakeWorker:
    def __init__(self, env_id, events):
        self.env_id, self.events, self.clock = env_id, events, 0
        self.terminal = False

    def send(self, method, payload):
        self.events.append(("send", self.env_id, method))
        self.method, self.payload = method, payload

    def receive(self, timeout):
        self.events.append(("receive", self.env_id, self.method))
        if self.method == "reset":
            self.clock = 0
            return {"env_id": self.env_id, "observation": {}, "info": {"reset": self.payload}}
        self.clock += len(self.payload["actions"])
        return {"env_id": self.env_id, "terminated": self.terminal, "truncated": False,
                "observation": {}, "info": {"clock": self.clock}}

    def close(self):
        pass


@pytest.fixture
def backend(config):
    backend = PAPParallelBackend.__new__(PAPParallelBackend)
    backend.config = config
    backend._broken = backend._closed = False
    backend._ready = [True, True]
    backend.lower, backend.upper = np.r_[[-3.]*6, 0.], np.r_[[3.]*6, .1]
    backend.events = []
    backend.workers = [FakeWorker(i, backend.events) for i in range(2)]
    backend.directory = Path("/unused/cpu-test")
    return backend


def test_dispatch_all_workers_before_waiting_and_indexed_reset(backend):
    command = np.zeros((2, 7)).tolist()
    backend.step([{"env_id": 0, "actions": command}, {"env_id": 1, "actions": command}])
    assert [event[:2] for event in backend.events] == [("send", 0), ("send", 1), ("receive", 0), ("receive", 1)]
    backend.events.clear()
    result = backend.reset([1], 2, {"by_env": {"1": {"color": "yellow"}}})
    assert backend.workers[0].clock == 2
    assert backend.workers[1].clock == 0
    assert result[0]["info"]["reset"]["color"] == "yellow"
    assert all(event[1] == 1 for event in backend.events)


def test_all_actions_validated_before_any_physics_mutation(backend):
    good = np.zeros((2, 7))
    bad = good.copy()
    bad[0, 2] = 99
    with pytest.raises(ValueError, match="no environment advanced"):
        backend.step([{"env_id": 0, "actions": good}, {"env_id": 1, "actions": bad}])
    assert backend.events == []
    with pytest.raises(ValueError):
        backend.reset([0, 1], 0, {"by_env": {"1": {"color": "purple"}}})
    assert backend.events == []


def test_terminal_environment_requires_reset(backend):
    backend.workers[0].terminal = True
    backend.step([{"env_id": 0, "actions": np.zeros((1, 7))}])
    with pytest.raises(ValueError, match="requires reset"):
        backend.step([{"env_id": 0, "actions": np.zeros((1, 7))}])
    assert backend._ready == [False, True]


def test_worker_failure_poisons_backend_without_retry(backend):
    def failure(timeout):
        raise TimeoutError("unknown physics state")
    backend.workers[0].receive = failure
    with pytest.raises(TimeoutError):
        backend.step([{"env_id": 0, "actions": np.zeros((1, 7))}])
    with pytest.raises(RuntimeError, match="state is unknown"):
        backend.reset([0], 1)


def test_authenticated_worker_process_ipc_and_indexed_time(backend, tmp_path, monkeypatch):
    """Exercise real process/IPC ownership, substituting only fake physics."""
    real_popen = pap.subprocess.Popen
    child = r'''
import json, os, sys
from multiprocessing.connection import Client
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
cfg = json.load(open(args['--config']))
c = Client((args['--host'], int(args['--port'])), authkey=bytes.fromhex(args['--auth']))
c.send_bytes(json.dumps({'ok': True, 'result': {'ready': True}}).encode())
clock = 0
while True:
    q = json.loads(c.recv_bytes())
    if q['method'] == 'close': break
    if q['method'] == 'reset': clock = 0
    else: clock += len(q['payload']['actions'])
    result = {'env_id': cfg['env_id'], 'observation': {}, 'terminated': False,
              'truncated': False, 'info': {'clock': clock, 'pid': os.getpid()}}
    c.send_bytes(json.dumps({'ok': True, 'result': result}).encode())
c.close()
'''
    def launch(args, **kwargs):
        return real_popen([args[0], "-c", child, *args[3:]], **kwargs)
    monkeypatch.setattr(pap.subprocess, "Popen", launch)
    backend.workers = []
    try:
        backend.workers = [pap._Worker(backend.config, i, tmp_path) for i in range(2)]
        initial = backend.reset([0, 1], 5)
        assert initial[0]["info"]["pid"] != initial[1]["info"]["pid"]
        backend.step([{"env_id": i, "actions": np.zeros((2, 7))} for i in range(2)])
        backend.reset([1], 6)
        rows = backend.step([{"env_id": i, "actions": np.zeros((1, 7))} for i in range(2)])
        assert [row["info"]["clock"] for row in rows] == [3, 1]
    finally:
        backend.close()


@pytest.fixture
def environment(config, monkeypatch):
    # Inject only the command container so state-machine logic runs without Kit.
    module = ModuleType("isaacsim.core.utils.types")
    module.ArticulationAction = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "isaacsim.core.utils.types", module)
    env = PAPEnvironment.__new__(PAPEnvironment)
    env.config = {**config, "env_id": 0}
    env.lower, env.upper = np.r_[[-3.]*6, 0.], np.r_[[3.]*6, .1]
    env.ready, env.ticks, env.max_ticks, env.subframes = True, 0, 12, 1
    env.active_object = 0
    env.contact_counts = {"link7": 0, "link8": 0}
    env.finger_contact = False
    env.last_sent = np.zeros(7)
    env.postprocessor = JointActionPostprocessor("joint_ema", .5)
    env.postprocessor.reset(np.zeros(7))
    env._state = lambda: env.last_sent.copy()
    env._observe = lambda: {"state": env._state().tolist(), "control_state": env.last_sent.tolist()}
    env.success_after = 1000
    env.outcome = SimpleNamespace(update=lambda *args, **kwargs: {"success": env.ticks >= env.success_after})
    cube = SimpleNamespace(get_world_pose=lambda: (np.array([.3, 0., .8]), np.array([1., 0., 0., 0.])),
        get_linear_velocity=lambda: np.zeros(3), get_angular_velocity=lambda: np.zeros(3))
    env.targets = []
    env.runtime = SimpleNamespace(world=SimpleNamespace(step=lambda **kwargs: None, render=lambda: None),
        robot=SimpleNamespace(num_dof=8), indices=np.arange(8), objects=[cube],
        controller=SimpleNamespace(apply_action=lambda value: env.targets.append(value["joint_positions"])))
    return env


def test_timeout_uses_actual_ticks_and_does_not_filter_gripper(environment):
    environment.max_ticks = 6  # 1.5 controls: terminal observation is at this tick.
    command = [[1., 0., 0., 0., 0., 0., .06]] * 3
    result = environment.step({"actions": command})
    assert result["truncated"] and not result["terminated"]
    assert result["info"]["physics_steps"] == 6
    assert result["info"]["elapsed_control_steps"] == 1.5
    assert result["discount"] == pytest.approx(.99**1.5)
    assert result["observation"]["control_state"] == [.75, 0., 0., 0., 0., 0., .06]
    assert np.allclose(environment.targets[0][6:], [.03, -.03])
    assert not environment.ready
    with pytest.raises(ValueError, match="reset"):
        environment.step({"actions": command})


def test_success_requires_bilateral_contacts_and_discounted_event(environment):
    environment.success_after = 3
    environment.contact_counts = {"link7": 1, "link8": 1}
    result = environment.step({"actions": np.zeros((4, 7))})
    assert result["terminated"] and result["info"]["success"]
    assert result["info"]["physics_steps"] == 3
    assert result["reward"] == pytest.approx(.99**.5)
    assert result["discount"] == pytest.approx(.99**.75)


def test_outcome_without_bilateral_grasp_is_not_success(environment):
    environment.success_after = 1
    environment.contact_counts = {"link7": 1, "link8": 0}
    result = environment.step({"actions": np.zeros((4, 7))})
    assert result["truncated"] and not result["terminated"]
    assert not result["info"]["success"]
    assert result["reward"] == 0


def test_dropped_object_is_failure_not_timeout(environment):
    environment.runtime.objects[0].get_world_pose = lambda: (np.array([.3, 0., .5]), np.array([1., 0., 0., 0.]))
    result = environment.step({"actions": np.zeros((4, 7))})
    assert result["terminated"] and not result["truncated"]
    assert result["info"]["termination_reason"] == "object_dropped"
    assert result["info"]["physics_steps"] == 1


def test_real_outcome_closed_gripper_is_json_serializable(environment):
    environment.outcome = PickPlaceOutcome(.03, .76, {
        "center_world": [.3, 0., .76], "inner_size_xy": [.1, .1],
        "floor_z_world": .765, "rim_z_world": .82, "yaw_deg": 0.})
    result = environment.step({"actions": np.zeros((1, 7))})
    assert result["info"]["physical_outcome"]["gripper_released"] is False
    assert json.loads(json.dumps(result))["info"]["success"] is False
