"""Task-only elevator tests without Isaac, a GPU, model services or sockets."""
from copy import deepcopy
import hashlib
import inspect
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pressb.runtime.contract import validate_metadata
from pressb.runtime.parallel import SharedWorldBackend
from pressb.tasks.elevator.native import NativeElevatorBackend
from pressb.tasks.elevator.parallel import ElevatorTask, classify_contact, panel_positions


@pytest.fixture
def task_config(tmp_path):
    snapshot = tmp_path / "scene.usda"
    snapshot.write_text("#usda 1.0\n")
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("<robot name='test'/>")
    scene = dict(robot_urdf=str(urdf), physics_dt=1/120, table_height=.76, home_q=[0.]*6,
        gripper_joint_positions_m=[.004,-.004], button_face_x=.4, button_bottom_z=.9,
        button_pitch_z=.04, button_column_y=.05, press_threshold=.0012, release_threshold=.0004,
        panel_offset_x_m=0., panel_offset_y_m=0., panel_randomization=dict(enabled=True,
            min_offset_x_m=-.01,max_offset_x_m=.01,min_offset_y_m=-.025,max_offset_y_m=.025))
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(scene))
    meta = tmp_path / "dataset/meta"
    meta.mkdir(parents=True)
    (meta / "collection_metadata.json").write_text(json.dumps(dict(config=scene,
        scene_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest())))
    return dict(config=str(cfg),snapshot=str(snapshot),dataset=str(meta.parent),asset_bundle=None,
                output=str(tmp_path / "new_run"),num_envs=2)


def test_native_entry_uses_the_common_backend_without_legacy_dispatch():
    assert issubclass(NativeElevatorBackend, SharedWorldBackend)
    assert NativeElevatorBackend.step is SharedWorldBackend.step
    assert NativeElevatorBackend.reset is SharedWorldBackend.reset
    assert NativeElevatorBackend.end_episode is SharedWorldBackend.end_episode
    assert "_LocalLegacyClient" not in inspect.getsource(NativeElevatorBackend)
    assert not any(name in ElevatorTask.__dict__ for name in ("step","reset","_step","_capture"))


def test_metadata_and_execution_keep_elevator_action_camera_and_time_contract(task_config):
    task = ElevatorTask(task_config)
    meta = task.metadata()
    validate_metadata(meta,"simulation")
    assert meta["action"]["representation"] == "pose9"
    assert len(meta["observation"]["state"]["names"]) == 8
    assert meta["control_state_dim"] == 37
    assert meta["task_features_dim"] == 12
    assert task.execution["terminal_boundary"] == "physics_tick"
    assert task.execution["physics_dt"] == 1/120
    assert task.execution["render_resolution"] == [320,240]
    assert task.execution["image_output_resolution"] == [224,224]
    assert task.execution["settling_ticks"] == 90


def test_full_resolution_and_64_slots_are_configuration_only(task_config):
    task = ElevatorTask({**task_config,"num_envs":64,"camera_resolution":640})
    assert task.execution["num_envs"] == 64
    assert task.execution["render_resolution"] == task.execution["image_output_resolution"] == [640,480]


def test_gpu_dynamics_rejected_instead_of_silently_ignored(task_config):
    with pytest.raises(ValueError,match="gpu_dynamics"):
        ElevatorTask({**task_config,"gpu_dynamics":True})


def test_unknown_expected_metadata_is_not_ignored(task_config):
    with pytest.raises(ValueError,match="metadata mismatch"):
        ElevatorTask({**task_config,"expected_metadata":{"old_contract":"something"}})


def test_reset_stream_matches_historical_seed_sequence(task_config):
    task = ElevatorTask(task_config)
    from pressb.tasks.elevator.backend import ElevatorBackend
    old = object.__new__(ElevatorBackend)
    old.floors, old.randomize_panel = task.floors, task.randomize_panel
    old._bounds, old._source = task.bounds, {"config":task.scene}
    for eid,seed in ((0,12),(1,999),(1,12)):
        actual = task.sample_reset(eid,seed,{})
        expected = old._episode(eid,seed,{})
        assert {key:actual[key] for key in expected} == expected
        assert actual["task"] == f"Press {expected['floor']} floor."


@pytest.mark.parametrize("options", [{"floor":23},{"floor":True},{"floor":24,"task":"Press 25 floor."},
    {"offset_x_m":.5},{"offset_y_m":float("nan")},{"unknown":0}])
def test_invalid_reset_options_do_not_mutate_task(task_config,options):
    task = ElevatorTask(task_config)
    before = task.metadata()
    with pytest.raises(ValueError):
        task.sample_reset(0,123,options)
    assert task.metadata() == before


def test_rest_positions_separate_world_origin_and_panel_translation(task_config):
    scene = ElevatorTask(task_config).scene
    base = panel_positions(scene,[0,0,0],0.,0.)
    translated = panel_positions(scene,[6,12,0],.01,-.02)
    np.testing.assert_allclose(translated-base,np.tile([6.01,11.98,0.],(12,1)))
    assert base[0,1] == .05 and base[6,1] == -.05
    assert base[5,2] == pytest.approx(.9+.04*5)


@pytest.mark.parametrize("paths,expected", [
    (("/World/Piper/link6","/World/Panel/Floor31/Cap","/World/Piper/link6/PressTip","/World/Panel/Floor31/Cap/Collision"),(7,None)),
    (("/World/Piper/link7","/World/Panel/Floor31/Cap","/World/Piper/link7/collisions","/World/Panel/Floor31/Cap/Collision"),(None,"non_stylus_button_contact")),
    (("/World/Piper/link3","/World/Piper/link4","a","b"),(None,"robot_self_contact")),
    (("/World/Piper/base_link","/World/Environment/Table","a","b"),(None,None)),
    (("/World/Piper/link6","/World/Piper/link7","/World/Piper/link6/PressStylus","/World/Piper/link7/collisions"),(None,None)),
    (("/World/Piper/link4","/World/Environment/Table","a","b"),(None,"robot_environment_contact")),
])
def test_contact_classification_preserves_stylus_rules(paths,expected):
    assert classify_contact(paths) == expected


class Buttons:
    def __init__(self, positions):
        self.positions = positions.reshape(-1,3).copy()
        self.orientation = np.tile([1.,0,0,0],(len(self.positions),1))
        self.velocity = np.zeros((len(self.positions),6))
    def get_world_poses(self):
        return self.positions.copy(),self.orientation.copy()
    def get_velocities(self):
        return self.velocity.copy()
    def set_world_poses(self,positions,orientations,indices):
        self.positions[indices],self.orientation[indices] = positions,orientations
    def set_velocities(self,velocity,indices):
        self.velocity[indices] = velocity


@pytest.fixture
def measured_task(task_config,monkeypatch):
    task = ElevatorTask(task_config)
    origin = np.asarray([[0.,0,0],[0,6.,0]])
    task.rests = np.stack([panel_positions(task.scene,x,0.,0.) for x in origin])
    task.buttons,task.button_rows = Buttons(task.rests),np.arange(24).reshape(2,12)
    task.forces = np.zeros((2,12))
    task.items = [dict(floor=floor,lit=set(),events=[],collisions=[],min_target_distance=1e6,
        max_tracking_error=0.,max_target_force=0.,max_target_travel=0.) for floor in (24,31)]
    transforms = np.tile(np.eye(4),(2,1,1))
    robot = SimpleNamespace(joint_arrays=lambda:(np.zeros((2,8)),np.zeros((2,8))),arm_dofs=np.arange(6),
        commands=np.zeros((2,8)),batched_controller=SimpleNamespace(fk=lambda q:transforms))
    task.engine = SimpleNamespace(num_envs=2,offsets=origin,robot=robot,
        template=SimpleNamespace(data={"base":[0,0,.76]}),
        backend=SimpleNamespace(ticks=np.asarray([1,1]),ready=np.asarray([True,True]),
            reset_specs=[{"floor":24},{"floor":31}]))
    task.light_changes = []
    monkeypatch.setattr(task,"_set_light",lambda *args:task.light_changes.append(args))
    return task


def test_target_success_requires_both_displacement_and_contact(measured_task):
    task = measured_task
    task.buttons.positions[0,0] += .002
    assert not task.after_tick([0])[0]["terminated"]
    task.forces[0,0] = .03
    result = task.after_tick([0])[0]
    assert result["success"] and result["terminated"] and not result["truncated"]
    assert result["reward"] == 1. and result["info"]["pressed_floors"] == [24]
    assert task.items[1]["events"] == []


def test_wrong_button_and_collision_precedence(measured_task):
    task = measured_task
    task.buttons.positions[1,0] += .002
    task.forces[0,1] = .03
    result = task.after_tick([0])[0]
    assert result["reason"] == "wrong_button_pressed" and not result["success"]
    task.items[0]["collisions"].append({"reason":"robot_environment_contact"})
    assert task.after_tick([0])[0]["reason"] == "unexpected_collision"


def test_timeout_remains_truncation_with_legacy_info_reason(measured_task):
    task = measured_task
    task.engine.backend.ticks[0] = round(task.config["max_seconds"]*120)
    row = task.after_tick([0])[0]
    assert row["truncated"] and not row["terminated"]
    assert row["info"]["termination"] == "time_limit"


def test_contact_impulse_is_tick_force_and_ended_slots_are_ignored(measured_task):
    task = measured_task
    paths = ("/World/Piper/link6","/World/Panel/Floor24/Cap","/World/Piper/link6/PressTip","/World/Panel/Floor24/Cap/Collision")
    task.on_contact(0,paths,[SimpleNamespace(impulse=[.001,0,0])])
    assert task.forces[0,0] == pytest.approx(.12)
    task.engine.backend.ready[0] = False
    task.on_contact(0,paths,[SimpleNamespace(impulse=[1,0,0])])
    assert task.forces[0,0] == pytest.approx(.12)
    task.forces[1,0] = 3.
    task.before_tick([0])
    assert task.forces[0,0] == 0 and task.forces[1,0] == 3.


def test_per_slot_observation_features(measured_task):
    values = measured_task.observation_fields([1,0])
    assert values[1]["task"] == "Press 31 floor."
    assert values[1]["task_features"][7] == 1.
    assert values[0]["task_features"][0] == 1.
    assert sum(values[1]["task_features"]) == 1.


def test_panel_reset_updates_selected_anchors_and_bodies_without_ticking(measured_task, monkeypatch):
    """A panel move must update its spring anchors and PhysX state together."""
    from pressb.tasks.elevator import dataset_scene
    task = measured_task
    task.after_settle()
    class Attribute:
        def __init__(self, value):
            self.value = deepcopy(value)
        def Set(self, value):
            self.value = deepcopy(value)
    attributes = {}
    def attribute(path, name):
        return attributes.setdefault((path, name), Attribute(None))
    task.engine.stage = SimpleNamespace(GetPrimAtPath=lambda path:
        SimpleNamespace(GetAttribute=lambda name: attribute(path, name)))
    shifts = []
    monkeypatch.setattr(dataset_scene, "_panel_shift", lambda *args: shifts.append(args[1:]))
    monkeypatch.setitem(sys.modules, "pxr", SimpleNamespace(Gf=SimpleNamespace(Vec3f=lambda *x: tuple(x))))
    task.source_offsets = [(0., 0.), (0., 0.)]
    task.panels = []
    for eid in range(2):
        anchors = [(Attribute(value), value.copy()) for value in task.rests[eid]]
        task.panels.append(dict(prefix=f"/World/envs/env_{eid}", anchors=anchors,
            transforms={floor: [("xformOp:translate", (0., 0., 0.))] for floor in range(24, 36)}))
    task.buttons.positions[1, 0] += .003
    task.buttons.velocity[1, 0] = .5
    task.forces[:] = 2.
    peer = deepcopy(task.diagnostic_state([0])[0])
    peer_item = deepcopy(task.items[0])
    spec = dict(floor=31, task="Press 31 floor.", offset_x_m=.01, offset_y_m=-.02)
    before_ticks = task.engine.backend.ticks.copy()
    result = task.reset_task([1], {1: spec})
    np.testing.assert_array_equal(task.engine.backend.ticks, before_ticks)
    assert task.diagnostic_state([0])[0] == peer and task.items[0] == peer_item
    assert shifts == [("/World/envs/env_1", 0., -.02, 0., .01)]
    np.testing.assert_allclose(task.buttons.positions[12:24], panel_positions(task.scene,[0,6,0],.01,-.02))
    for changed, initial in task.panels[1]["anchors"]:
        np.testing.assert_allclose(changed.value, initial+[.01,-.02,0.])
    for untouched, initial in task.panels[0]["anchors"]:
        np.testing.assert_array_equal(untouched.value, initial)
    assert task.forces[0,0] == 2. and np.all(task.forces[1] == 0.)
    assert all(eid == 1 and on is False for eid, floor, on in task.light_changes)
    assert result[1]["reset_physics_steps"] == 0
