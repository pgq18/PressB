"""CPU reset-contract checks; no simulator or GPU is started."""
from copy import deepcopy
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from pressb.tasks.elevator.backend import ElevatorBackend
from pressb.tasks.elevator.fast_simulation import FastIsaacVectorBackend, IndexedSimulationService
from test_rl_nodes_elevator import LocalClient, PhysicsFixture, actions


class FourPhysics(PhysicsFixture):
    def health(self):
        return {**super().health(), "num_envs": 4}

    def reset(self, episodes, seed):
        self.calls.append(("reset", deepcopy(episodes), seed))
        self.episodes.update(enumerate(episodes))
        return [self.observe(i) for i in range(4)]


def adapter():
    client = LocalClient()
    client.physics = FourPhysics()
    client.service = IndexedSimulationService(client.physics, single_gamma=.995)
    return ElevatorBackend({"endpoint": "http://fixture"}, client=client), client


def test_unsorted_subset_batch_preserves_singleton_seed_layouts_and_active_peers():
    batched, batch_client = adapter()
    sequential, sequential_client = adapter()
    for backend, client in ((batched, batch_client), (sequential, sequential_client)):
        backend.reset([0, 1, 2, 3], seed=10)
        client.physics.events = {1:(True, 5), 3:(True, 9)}
        backend.step(actions([0, 1, 2, 3]))
    peers = {i:deepcopy(batch_client.service.items[i]) for i in (0, 2)}
    options = {"by_env":{"1":{"floor":31}, "3":{"task":"Press 25 floor."}}}
    rows = batched.reset([3, 1], seed=999, options=options, env_seeds={"1":123, "3":987})
    sequential.reset([3], seed=987, options={"task":"Press 25 floor."})
    sequential.reset([1], seed=123, options={"floor":31})
    assert [row["env_id"] for row in rows] == [3, 1]
    for i in (1, 3):
        expected = sequential_client.physics.episodes[i]
        actual = batch_client.physics.episodes[i]
        assert {key:actual[key] for key in expected} == expected
        assert actual["seed"] == {1:123, 3:987}[i]
    assert all(batch_client.service.items[i] == peers[i] for i in peers)
    assert len([row for row in batch_client.physics.calls if row[0] == "reset_envs"]) == 1
    assert len([row for row in sequential_client.physics.calls if row[0] == "reset_envs"]) == 2
    assert "per_env_reset_seeds" in batched.describe()["capabilities"]


def test_initial_per_environment_seed_layout_is_order_independent():
    first, ca = adapter()
    second, cb = adapter()
    seeds = {0:11, 1:22, 2:33, 3:44}
    first.reset([3, 1, 0, 2], 0, env_seeds=seeds)
    second.reset([0, 1, 2, 3], 999, env_seeds={str(i):seed for i,seed in seeds.items()})
    assert ca.physics.episodes == cb.physics.episodes
    for i, seed in seeds.items():
        assert ca.physics.episodes[i] == {**first._episode(i, seed, {}), "seed":seed}


def test_shared_seed_requests_keep_legacy_episode_payload_unchanged():
    backend, client = adapter()
    backend.reset([0, 1, 2, 3], 45)
    assert client.physics.episodes == {i:backend._episode(i,45,{}) for i in range(4)}
    assert all("seed" not in row for row in client.physics.episodes.values())


@pytest.mark.parametrize("seed_map", [{0:1}, {0:1,1:2,2:3,3:4,4:5},
    {"0":1,0:2,1:3,2:4,3:5}, {0:1,1:True,2:3,3:4}, {0:1,1:2,2:3,3:-1}])
def test_invalid_seed_map_rejected_before_any_reset(seed_map):
    backend, client = adapter()
    with pytest.raises(ValueError):
        backend.reset([0,1,2,3], 42, env_seeds=seed_map)
    assert not client.physics.calls and backend._fault is None


@pytest.mark.parametrize("individual", [False, True])
def test_actual_fast_reset_captures_once_and_keeps_peer_physics(tmp_path, monkeypatch, individual):
    """Exercise production reset_envs with tensor/scene fixtures, not a rewrite."""
    from pressb.tasks.elevator import dataset_scene
    backend = object.__new__(FastIsaacVectorBackend)
    q = np.arange(24, dtype=float).reshape(3,8)
    v = q / 10
    before_q, before_v = q.copy(), v.copy()
    settled_q = q + 100
    settled_v = v + 10
    backend.num_envs = 3
    backend.home = np.zeros(6)
    backend.cfg = {"gripper_joint_positions_m":[.004,.004]}
    backend.window = 1
    backend.output = tmp_path
    backend._run_id = "fixture"
    backend._reset_serial = 0
    backend._commands = np.zeros((3,8))
    backend.arm_rows = np.arange(3)
    backend.arm_dofs = np.arange(6)
    backend.finger_dofs = np.array([6,7])
    backend.button_rows = np.arange(36).reshape(3,12)
    backend._settled = dict(q=settled_q, velocity=settled_v,
        button_relative=np.zeros((3,12,3)), button_orientation=np.tile([1,0,0,0],(3,12,1)),
        button_velocity=np.zeros((3,12,6)))
    peer = {"done":False,"identity":"unchanged-peer"}
    backend._items = [{"done":True}, peer, {"done":True}]
    backend.recording = {1}
    backend.forces = np.ones(3)
    backend.collisions = [["old"], ["peer-contact"], ["old"]]
    backend.envs = [SimpleNamespace(button_info={f:{"body_center":[0,0,0]} for f in range(24,36)}) for _ in range(3)]
    backend._check_thread = backend._check_frozen_state = backend._remember_frozen_state = lambda:None
    backend._joint_arrays = lambda:(q.copy(),v.copy())
    backend.arms = SimpleNamespace(
        set_joint_positions=lambda values,indices:q.__setitem__(indices,values),
        set_joint_velocities=lambda values,indices:v.__setitem__(indices,values),
        set_joint_position_targets=lambda *args,**kwargs:None)
    backend.button_view = SimpleNamespace(set_world_poses=lambda *a,**k:None,
                                         set_velocities=lambda *a,**k:None)
    backend.batched_controller = SimpleNamespace(reset=lambda *args:None)
    backend.world = SimpleNamespace(current_time=123.,physics_sim_view=SimpleNamespace(
        update_articulations_kinematic=lambda:None))
    monkeypatch.setattr(dataset_scene,"set_panel_offset",lambda world,env,y,x:None)
    monkeypatch.setattr(dataset_scene,"validate_panel_layout",lambda world,env:{"fixture":True})
    omni = ModuleType("omni")
    physx = ModuleType("omni.physx")
    physx.get_physx_interface = lambda:SimpleNamespace(update_transformations=lambda *args:None)
    omni.physx = physx
    monkeypatch.setitem(sys.modules,"omni",omni)
    monkeypatch.setitem(sys.modules,"omni.physx",physx)
    captures=[]
    backend._capture = lambda **kwargs:captures.append(kwargs) or "fixture-images"
    backend._refresh_metrics = lambda:None
    backend._encode_images = lambda ids,tiles:{int(i):{} for i in ids}
    backend._measure_metrics = lambda i:None
    backend._observation = lambda i,*args:{"state":[i]}
    backend._info = lambda i:{}
    backend.timing = {"reset_calls":0,"total_reset_seconds":0.}
    episodes={2:{"floor":25,"offset_x_m":.01,"offset_y_m":.02},
              0:{"floor":31,"offset_x_m":-.01,"offset_y_m":-.02}}
    if individual:
        episodes[2]["seed"],episodes[0]["seed"]=987,123
    result=backend.reset_envs(episodes,19)
    assert [row["env_id"] for row in result]==[0,2]
    assert captures==[{"changed":True}] and backend.world.current_time==123.
    np.testing.assert_array_equal(q[1],before_q[1])
    np.testing.assert_array_equal(v[1],before_v[1])
    np.testing.assert_array_equal(q[[0,2]],settled_q[[0,2]])
    assert backend._items[1] is peer and backend.collisions[1]==["peer-contact"]
    for i in (0,2):
        expected=episodes[i].get("seed",19)
        assert backend._items[i]["seed"]==expected
        receipt=json.loads((tmp_path/f"indexed_reset_000001/env_{i:03d}_reset.json").read_text())
        assert receipt["seed"]==expected and receipt["episode"]==episodes[i]
        assert receipt["info"]["reset_physics_steps"]==0
