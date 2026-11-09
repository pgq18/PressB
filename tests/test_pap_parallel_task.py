"""PAP outcome/reset semantics with the shared engine's measured-state hooks."""
from copy import deepcopy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pressb.tasks.pap.parallel import PAPTask


class Bodies:
    def __init__(self):
        self.positions = np.array([[6.3, .02, .775], [.3, .02, .775]])
        self.quaternions = np.tile([1., 0., 0., 0.], (2, 1))
        self.velocities = np.ones((2, 6))

    def get_world_poses(self):
        return self.positions.copy(), self.quaternions.copy()

    def get_velocities(self):
        return self.velocities.copy()

    def set_world_poses(self, positions, rotations, rows):
        self.positions[rows], self.quaternions[rows] = positions, rotations

    def set_velocities(self, values, rows):
        self.velocities[rows] = values


def task():
    obj = object.__new__(PAPTask)
    geometry = {"shape": "cube", "edge": .029, "size_xyz": [.029]*3}
    obj.config = dict(objects=[{"geometry": geometry}], physics_dt=.1,
        scene={"table_height": .76}, outcome={"stable_seconds": .5, "lift_height": .04},
        settling_ticks=24, terminate_on_drop=True, drop_below_table_m=.05,
        reward={"success": 1., "failure": -.5})
    obj.bodies, obj.object_rows = Bodies(), np.array([1, 0])
    robot = SimpleNamespace(observations=lambda ids: {i: {"state": [0.]*6+[.05]} for i in ids},
        postprocessors=[SimpleNamespace(summary=lambda: {"mode": "none"}) for _ in range(2)])
    obj.engine = SimpleNamespace(backend=SimpleNamespace(ready=np.ones(2, dtype=bool), ticks=np.array([2, 3])),
        robot=robot, offsets=np.array([[0., 0., 0.], [6., 0., 0.]]),
        template=SimpleNamespace(robot_path="/World/Piper"))
    obj.source_scene = {"box_inside": {"origin_world": [.3, .2, .76], "yaw_deg": 0.,
        "min_local": [-.05, -.075, 0.], "max_local": [.05, .075, .06]}}
    obj.source_base = np.array([0., 0., .76])
    obj.object_path = "/World/PickPlace/Object"
    obj.material_paths = [f"/World/envs/env_{i}/Looks/Object" for i in range(2)]
    obj.contact_counts = np.array([[3, 4], [5, 6]])
    obj.finger_contact = np.ones(2, dtype=bool)
    obj.outcomes = [SimpleNamespace(update=lambda *a, **kw: {"success": True}),
                    SimpleNamespace(update=lambda *a, **kw: {"success": False})]
    obj.after_settle()
    return obj


def test_contact_requires_object_and_finger_of_active_slot():
    obj = task()
    obj.contact_counts[:] = 0
    obj.before_tick([0, 1])
    obj.on_contact(0, [obj.object_path, "/World/Piper/link7"], [object()])
    obj.on_contact(0, [obj.object_path, "/World/Piper/link8"], [object()])
    obj.on_contact(1, [obj.object_path, "/World/Piper/link6"], [object()])
    obj.on_contact(1, ["/World/Table", "/World/Piper/link7"], [object()])
    obj.on_contact(1, [obj.object_path, "/World/Piper/link7"], [])
    obj.engine.backend.ready[1] = False
    obj.on_contact(1, [obj.object_path, "/World/Piper/link7"], [object()])
    assert obj.contact_counts.tolist() == [[1, 1], [0, 0]]
    assert obj.finger_contact.tolist() == [True, False]
    obj.before_tick([0])
    assert not obj.finger_contact.any()
    assert obj.contact_counts.tolist() == [[1, 1], [0, 0]]


def test_success_requires_bilateral_history_and_preserves_measured_rows():
    obj = task()
    result = obj.after_tick([1, 0])
    assert result[0]["terminated"] and result[0]["success"]
    assert result[0]["reward"] == 1.
    assert result[0]["info"]["object_position_world"] == [.3, .02, .775]
    assert result[1]["info"]["object_position_world"] == [6.3, .02, .775]
    assert not result[1]["terminated"]
    obj.contact_counts[0, 1] = 0
    result = obj.after_tick([0])[0]
    assert not result["terminated"] and not result["success"]
    assert result["reward"] == 0
    assert result["info"]["physical_outcome"]["success"]


def test_drop_event_is_task_owned_but_timeout_and_discount_are_runtime_owned():
    obj = task()
    obj.bodies.positions[0, 2] = .5
    result = obj.after_tick([1])[1]
    assert result["terminated"] and not result["truncated"]
    assert result["reason"] == "object_dropped" and result["reward"] == -.5
    obj.config["terminate_on_drop"] = False
    obj.engine.backend.ticks[1] = 100000
    result = obj.after_tick([1])[1]
    assert not result["terminated"] and not result["truncated"]
    assert result["reward"] == 0


def test_outcome_receives_exact_episode_time_gripper_and_contact():
    obj = task()
    calls = []
    def record(*args, **kwargs):
        calls.append((deepcopy(args), deepcopy(kwargs)))
        return {"success": False}
    obj.outcomes[0] = SimpleNamespace(update=record)
    obj.after_tick([0])
    args, kwargs = calls[0]
    assert args[0] == pytest.approx(.2)
    assert args[-1] == .05
    assert kwargs == {"finger_contact": True}
    assert np.array_equal(args[1], [.3, .02, .775])


@pytest.mark.parametrize("field", ["positions", "quaternions", "velocities"])
def test_nonfinite_task_physics_rejected(field):
    obj = task()
    getattr(obj.bodies, field)[0, 0] = np.nan
    with pytest.raises(RuntimeError, match="Nonfinite"):
        obj.after_tick([0, 1])


def test_selected_task_reset_keeps_peer_body_and_contact_history(monkeypatch):
    obj = task()
    material_writes = []
    obj.engine.stage = SimpleNamespace(GetPrimAtPath=lambda path: path)
    shader = lambda path: SimpleNamespace(GetInput=lambda name: SimpleNamespace(
        Set=lambda value: material_writes.append((path, name, value))))
    monkeypatch.setitem(sys.modules, "pxr", SimpleNamespace(Gf=SimpleNamespace(Vec3f=lambda *v: tuple(v)),
                                                        UsdShade=SimpleNamespace(Shader=shader)))
    before = obj.diagnostic_state([0])[0]
    peer_outcome = obj.outcomes[0]
    spec = dict(center_xy=[.4, -.06], yaw_deg=90., color_rgb=[0., 1., 0.])
    rows = obj.reset_task([1], {1: spec})
    assert obj.diagnostic_state([0])[0] == before
    assert obj.outcomes[0] is peer_outcome
    assert obj.contact_counts.tolist() == [[3, 4], [0, 0]]
    assert obj.finger_contact.tolist() == [True, False]
    assert np.allclose(obj.bodies.positions[0], [6.4, -.06, .775])
    assert np.allclose(obj.bodies.quaternions[0], [2**-.5, 0., 0., 2**-.5])
    assert np.array_equal(obj.bodies.velocities[0], np.zeros(6))
    assert np.array_equal(obj.bodies.velocities[1], np.ones(6))
    assert obj.outcomes[1].box["center_world"] == [6.3, .2, .76]
    assert rows[1]["settling_physics_steps"] == 0
    assert len(material_writes) == 2
    assert all("env_1/" in path for path, _, _ in material_writes)
    assert np.array_equal(obj.engine.backend.ticks, [2, 3])


def test_reset_outcome_after_physical_lift_is_strict_json(monkeypatch):
    # Clone offsets are NumPy values. Keep them out of outcome scalar state:
    # after a positive physical lift, np.float64 tabletop_z otherwise turns
    # max_bottom_clearance/lifted into NumPy scalars that break RPC JSON.
    import json
    obj = task()
    obj.engine.stage = SimpleNamespace(GetPrimAtPath=lambda path: path)
    shader = lambda path: SimpleNamespace(GetInput=lambda name: SimpleNamespace(Set=lambda value: None))
    monkeypatch.setitem(sys.modules, "pxr", SimpleNamespace(Gf=SimpleNamespace(Vec3f=lambda *v: tuple(v)),
                                                        UsdShade=SimpleNamespace(Shader=shader)))
    spec = dict(center_xy=[.4, -.06], yaw_deg=0., color_rgb=[0., 1., 0.])
    obj.reset_task([0, 1], {0: spec, 1: spec})
    obj.bodies.positions[:, 2] = .9
    rows = obj.after_tick([0, 1])
    json.dumps(rows, allow_nan=False)
    for eid in (0, 1):
        outcome = rows[eid]["info"]["physical_outcome"]
        assert type(obj.outcomes[eid].tabletop_z) is float
        assert type(outcome["lifted"]) is bool and outcome["lifted"]
        assert type(outcome["max_bottom_clearance_m"]) is float
        assert outcome["max_bottom_clearance_m"] == pytest.approx(.9-.029/2-.76)
