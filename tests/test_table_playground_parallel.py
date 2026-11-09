"""CPU tests for task hooks with indexed measured-body views.

These validate task state and reset isolation, not PhysX collision dynamics.
The production shared scheduler is tested separately in test_parallel_backend.
"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from pressb.tasks.playground.parallel import PlaygroundTask
from pressb.tasks.table_buttons.parallel import TableButtonsTask
from pressb.tasks.table_buttons.task import COLORS
from pressb.tasks.table_buttons.backend import load_config as table_config
from pressb.tasks.playground.backend import load_config as playground_config


class Attribute:
    def __init__(self, value):
        self.value = tuple(value)

    def Get(self):
        return self.value

    def Set(self, value):
        self.value = tuple(value)


class Bodies:
    def __init__(self, positions):
        self.positions = np.asarray(positions, dtype=float)
        self.rotations = np.tile([1., 0., 0., 0.], (len(positions), 1))
        self.velocities = np.arange(len(positions)*6, dtype=float).reshape(-1, 6) + 1
        self.writes = []
        self.velocity_writes = []

    def get_world_poses(self):
        return self.positions.copy(), self.rotations.copy()

    def get_velocities(self):
        return self.velocities.copy()

    def set_world_poses(self, positions, rotations, rows):
        self.writes.append(list(rows))
        self.positions[rows], self.rotations[rows] = positions, rotations

    def set_velocities(self, values, rows):
        self.velocity_writes.append(list(rows))
        self.velocities[rows] = values


def table_task(**config):
    task = TableButtonsTask({"num_envs": 2, "randomization": {"positions": False}, **config})
    offsets = np.array([[0., 0., 0.], [6., -6., 0.]])
    buttons, initial = {}, []
    for color in COLORS:
        xy = task.config["scene"]["buttons"][color]["center_xy"]
        path = f"/World/TableButtons/Buttons/{color}"
        rest = [*xy, .80]
        buttons[color] = dict(body_path=path + "/Cap", joint_path=path + "/Spring",
            center_xy_base=xy, cap_rest_world=rest)
        initial.extend([rest, [*xy, .78], [*xy, .79]])
    positions = np.vstack([np.asarray(initial)+offset for offset in offsets])
    # Tensor views need not enumerate environments in numerical order.
    rows = np.arange(18).reshape(2, 9)[:, ::-1]
    bodies = Bodies(np.empty_like(positions))
    bodies.positions[rows.reshape(-1)] = positions
    anchors = {f"env{eid}{button['joint_path']}": Attribute(np.asarray(button["cap_rest_world"])+offsets[eid])
        for eid in range(2) for button in buttons.values()}
    engine = SimpleNamespace(num_envs=2, offsets=offsets,
        template=SimpleNamespace(data={"scene": {"buttons": buttons}}, robot_path="/World/Piper"),
        stage=SimpleNamespace(GetPrimAtPath=lambda path: SimpleNamespace(GetAttribute=lambda name: anchors[path])),
        relocate=lambda eid, path: f"env{eid}{path}",
        backend=SimpleNamespace(ready=np.ones(2, dtype=bool), ticks=np.array([4, 8]), reset_specs=[None]*2),
        rigid_bodies=lambda paths: (bodies, rows), world=SimpleNamespace(current_time=3.))
    task.bind(engine)
    task.after_settle()
    specs = {eid: task.sample_reset(eid, 12, {"target_color": "red"}) for eid in range(2)}
    engine.backend.reset_specs[:] = [deepcopy(specs[eid]) for eid in range(2)]
    task.reset_task([0, 1], specs)
    bodies.writes.clear()
    bodies.velocity_writes.clear()
    return task


def test_table_indexed_reset_moves_all_fixture_parts_and_three_world_anchors_without_touching_peer():
    task = table_task()
    task.bodies.velocities[:] = 1.
    task.wrong_contact[0] = True
    peer_checker = task.checkers[0]
    before = task.diagnostic_state([0])[0]
    original = task.initial_positions[1].copy()
    spec = deepcopy(task.engine.backend.reset_specs[1])
    deltas = dict(zip(COLORS, ([.01, -.02], [-.02, .03], [.04, .01])))
    for color in COLORS:
        spec["positions"][color] = (np.asarray(spec["positions"][color]) + deltas[color]).tolist()
    task.wrong_contact[1] = True
    task.blocking_contacts[1] = [{"force_n": 3.}]
    info = task.reset_task([1], {1: spec})
    assert task.diagnostic_state([0])[0] == before
    assert task.checkers[0] is peer_checker
    assert not task.wrong_contact[1] and not task.blocking_contacts[1]
    assert task.bodies.writes == [task.rows[1].tolist()]
    for j, color in enumerate(COLORS):
        selected = task.button_rows[color]
        delta = [*deltas[color], 0.]
        np.testing.assert_allclose(task.bodies.positions[task.rows[1, selected]], original[selected]+delta)
        np.testing.assert_allclose(task._anchor_attribute(1, color).Get(), task.rest_world[1, j])
        np.testing.assert_allclose(task.rest_world[1, j], original[selected[0]]+delta)
    cap_rows = [task.rows[1, task.button_rows[color][0]] for color in COLORS]
    fixture_rows = [row for row in task.rows[1] if row not in cap_rows]
    assert task.bodies.velocity_writes == [cap_rows]
    np.testing.assert_array_equal(task.bodies.velocities[cap_rows], np.zeros((3, 6)))
    np.testing.assert_array_equal(task.bodies.velocities[fixture_rows], np.ones((6, 6)))
    assert info[1]["zero_tick_reset"]
    assert task.engine.world.current_time == 3.
    assert task.engine.backend.ticks.tolist() == [4, 8]


def contact(task, eid, *, color="red", link="link7", force=.2, collider_only=False):
    cap = task.buttons[color]["body_path"]
    robot = f"/World/Piper/{link}"
    actors = [cap, robot, cap+"/collision", robot+"/collision"]
    if collider_only:
        actors = ["/World/UnknownBody0", "/World/UnknownBody1", cap+"/collision", robot+"/collision"]
    task.on_contact(eid, actors, [SimpleNamespace(impulse=[0., 0., force*task.config["physics_dt"]])])


@pytest.mark.parametrize("depth,force", [(0., .2), (.0032, 0.)])
def test_table_press_needs_both_measured_displacement_and_force(depth, force):
    task = table_task()
    row = task.rows[0, task.button_rows["red"][0]]
    task.bodies.positions[row, 2] -= depth
    for tick in range(1, 9):
        task.before_tick([0])
        task.engine.backend.ticks[0] = tick
        contact(task, 0, force=force)
        result = task.after_tick([0])[0]
    assert not result["success"] and not result["terminated"] and result["reward"] == 0


def test_table_contact_routes_colliders_and_keeps_checks_after_latched_success():
    task = table_task(scene_overrides={"success": {"hold_seconds": 0.}})
    row = task.rows[1, task.button_rows["red"][0]]
    task.bodies.positions[row, 2] -= .0032
    task.before_tick([0, 1])
    contact(task, 1, collider_only=True)
    first = task.after_tick([0, 1])
    assert first[1]["success"] and first[1]["reward"] == 1.
    assert not first[0]["success"]
    task.engine.backend.ticks[1] += 1
    task.before_tick([1])
    contact(task, 1, color="yellow", force=.03)
    later = task.after_tick([1])[1]
    assert later["info"]["physical_outcome"]["success"]
    assert later["terminated"] and not later["success"] and later["reward"] == 0.
    assert later["reason"] == "non_target_finger_contact"


@pytest.mark.parametrize("policy,expected", [("finger_closure_dual", 0), ("collector_strict", 1)])
def test_table_only_exact_finger_pair_is_exempt_under_dual_policy(policy, expected):
    task = table_task(collision_policy=policy)
    contacts = [SimpleNamespace(impulse=[0., 0., .003])]
    task.on_contact(0, ["/World/Piper/link7", "/World/Piper/link8",
        "/World/Piper/link7/collision", "/World/Piper/link8/collision"], contacts)
    assert len(task.blocking_contacts[0]) == expected
    task.on_contact(1, ["/World/Piper/link6", "/World/Piper/link8",
        "/World/Piper/link6/collision", "/World/Piper/link8/collision"], contacts)
    assert len(task.blocking_contacts[1]) == 1
    # A fingertip hitting the housing is a blocking contact, not a valid press.
    task.on_contact(1, [task.buttons["red"]["body_path"].replace("Cap", "Housing"),
        "/World/Piper/link7", "", ""], contacts)
    assert len(task.blocking_contacts[1]) == 2


def test_table_contacts_ignore_inactive_environment_and_nonfinger_cap_is_blocking():
    task = table_task()
    task.engine.backend.ready[0] = False
    contact(task, 0)
    contact(task, 1, link="link6")
    assert not any(task.contact_forces[0].values())
    assert not any(task.contact_forces[1].values())
    assert len(task.blocking_contacts[1]) == 1


def playground_task(objects=None):
    objects = [] if objects is None else objects
    task = PlaygroundTask({"num_envs": 2, "scene_overrides": {"objects": objects}})
    paths = {spec["name"]: {"path": f"/World/Playground/Objects/{spec['name']}"} for spec in objects}
    positions = [[6.+spec["center_xy"][0], spec["center_xy"][1], .775] for spec in objects]
    positions += [[spec["center_xy"][0], spec["center_xy"][1], .775] for spec in objects]
    bodies = Bodies(positions)
    rows = np.arange(len(objects)*2).reshape(2, len(objects))[::-1]
    def view(request):
        assert request, "Empty playground must not construct a rigid body view"
        return bodies, rows
    engine = SimpleNamespace(num_envs=2, template=SimpleNamespace(data={"scene": {"objects": paths}}),
        rigid_bodies=view, world=SimpleNamespace(current_time=10.),
        backend=SimpleNamespace(ticks=np.array([13, 17])))
    task.bind(engine)
    task.after_settle()
    return task


OBJECTS = [{"name": "cube", "size": [.029]*3, "center_xy": [.35, -.02], "yaw_deg": 10.}]


def test_playground_indexed_object_reset_restores_pose_velocity_and_preserves_peer():
    task = playground_task(OBJECTS)
    peer = task.diagnostic_state([0])[0]
    task.bodies.positions[0] += [.1, -.2, .4]
    spec = task.sample_reset(1, 4, {"objects": {"cube": {"center_xy": [.4, .08], "yaw_deg": 90.}}})
    info = task.reset_task([1], {1: spec})
    np.testing.assert_allclose(task.bodies.positions[0], [6.4, .08, .775])
    np.testing.assert_allclose(task.bodies.rotations[0], [2**-.5, 0., 0., 2**-.5])
    np.testing.assert_array_equal(task.bodies.velocities[0], np.zeros(6))
    assert task.diagnostic_state([0])[0] == peer
    assert task.engine.backend.ticks.tolist() == [13, 17]
    assert task.engine.world.current_time == 10.
    assert not info[1]["task_outcome_available"]


def test_empty_playground_never_constructs_object_view_or_invents_success():
    task = playground_task()
    specs = {eid: task.sample_reset(eid, 4, {}) for eid in (0, 1)}
    task.reset_task([0, 1], specs)
    assert task.bodies is None
    assert task.diagnostic_state([0, 1]) == {0: {"objects": {}}, 1: {"objects": {}}}
    for outcome in task.after_tick([0, 1]).values():
        assert outcome["reward"] == 0 and not outcome["success"] and not outcome["terminated"]
        assert "truncated" not in outcome  # Episode time is enforced by shared runtime.


@pytest.mark.parametrize("factory", [table_task, lambda: playground_task(OBJECTS)])
@pytest.mark.parametrize("field", ["positions", "rotations", "velocities"])
def test_nonfinite_measured_task_body_state_is_not_a_valid_transition(factory, field):
    task = factory()
    getattr(task.bodies, field)[0, 0] = np.nan
    with pytest.raises(RuntimeError, match="Nonfinite"):
        task.after_tick([0, 1])


@pytest.mark.parametrize("objects", [
    {"unknown": {}}, {"cube": {"center_xy": [2., 0.]}}, {"cube": {"yaw_deg": float("nan")}},
    {"cube": {"size": [.1]*3}}, {"cube": {"center_xy": [.3, float("nan")]}}
])
def test_playground_invalid_reset_geometry_rejected_without_body_mutation(objects):
    task = playground_task(OBJECTS)
    before = task.diagnostic_state([0, 1])
    with pytest.raises(ValueError):
        task.sample_reset(1, 42, {"objects": objects})
    assert task.diagnostic_state([0, 1]) == before
    assert task.bodies.writes == []


@pytest.mark.parametrize("task_type", [TableButtonsTask, PlaygroundTask])
def test_parallel_task_supports_64_slots_and_shared_execution_config(task_type):
    config = {"num_envs": 64, "spacing_m": 7., "render_resolution": [320, 240],
        "observation_render_frames": 3, "reset_render_frames": 4, "png_workers": 2}
    if task_type is TableButtonsTask:
        config["randomization"] = {"positions": False}
    task = task_type(config)
    assert task.metadata()["num_envs"] == task.execution["num_envs"] == 64
    assert task.execution["terminal_boundary"] == "control_interval"
    assert task.execution["spacing_m"] == 7.
    assert task.execution["render_resolution"] == [320, 240]
    assert task.execution["observation_render_frames"] == 3
    assert task.execution["reset_render_frames"] == 4


@pytest.mark.parametrize("module,task_type", [
    ("pressb.tasks.table_buttons.backend", TableButtonsTask),
    ("pressb.tasks.playground.backend", PlaygroundTask),
])
def test_default_backend_factory_delegates_to_shared_runtime(monkeypatch, module, task_type):
    import importlib
    import pressb.runtime.parallel
    captured = []
    def factory(task, *, engine_factory):
        captured.append((task, engine_factory))
        return "shared runtime"
    monkeypatch.setattr(pressb.runtime.parallel, "SharedWorldBackend", factory)
    config = {"num_envs": 2}
    if task_type is TableButtonsTask:
        config["randomization"] = {"positions": False}
    assert importlib.import_module(module).create_backend(config) == "shared runtime"
    assert len(captured) == 1 and isinstance(captured[0][0], task_type)
    assert captured[0][1] is None


def test_table_prepares_native_collision_libraries_and_profile_cache_before_engine(monkeypatch):
    import pressb.tasks.table_buttons.placement
    calls = []
    class Placement:
        def __init__(self, kin, cfg):
            self.profile = cfg["position_sampling"]["spacing_profile"]
            calls.append(("prepare", self.profile))
        def sample(self, seed):
            calls.append(("sample", self.profile, seed))
            return {"positions": {"red": [.33, -.04], "yellow": [.40, 0.], "blue": [.33, .04]}}
    monkeypatch.setattr(pressb.tasks.table_buttons.placement, "ReachablePlacement", Placement)
    task = TableButtonsTask({"num_envs": 2, "randomization": {"profiles": ["near"]}})
    assert calls == [("prepare", "near")]
    assert not hasattr(task, "engine")
    task.sample_reset(1, 42, {})
    task.sample_reset(1, 43, {})
    assert sum(call[0] == "prepare" for call in calls) == 1


def test_table_environment_seed_streams_are_independent_and_reproducible():
    def sampler(profile, seed):
        offset = float(np.random.default_rng(seed).uniform(-.005, .005))
        return {"positions": {"red": [.33+offset, -.04], "yellow": [.40+offset, 0.], "blue": [.33+offset, .04]}}
    task = TableButtonsTask({"num_envs": 2, "randomization": {"profiles": ["near"]}}, layout_sampler=sampler)
    first = task.sample_reset(0, 42, {})
    second = task.sample_reset(1, 42, {})
    assert first == task.sample_reset(0, 42, {})
    assert second == task.sample_reset(1, 42, {})
    assert first["seed"] == second["seed"] == 42
    assert first["layout_seed"] != second["layout_seed"]
    assert first["positions"] != second["positions"]


@pytest.mark.parametrize("load", [table_config, playground_config])
def test_render_resolution_preserves_camera_aspect_but_square_transport_output_is_allowed(load):
    original = load({})
    for resolution in ([320, 240], [1280, 960]):
        cfg = load({"render_resolution": resolution, "image_output_resolution": [224, 224]})
        assert cfg["render_resolution"] == resolution
        assert cfg["image_output_resolution"] == [224, 224]
        assert cfg["scene"]["up_camera"] == original["scene"]["up_camera"]
        assert cfg["scene"].get("wrist_camera") == original["scene"].get("wrist_camera")
    with pytest.raises(ValueError, match="up camera aspect ratio"):
        load({"render_resolution": [224, 224]})


@pytest.mark.parametrize("load", [table_config, playground_config])
def test_wrist_sensor_aspect_is_checked_independently_of_global_camera(load):
    with pytest.raises(ValueError, match="wrist camera aspect ratio"):
        load({"scene_overrides": {"wrist_camera": {"resolution": [640, 360]}}})
    # Explicit new calibration with matching sensors may use another aspect.
    cfg = load({"render_resolution": [320, 180], "scene_overrides": {
        "up_camera": {"resolution": [640, 360]}, "wrist_camera": {"resolution": [640, 360]}}})
    assert cfg["render_resolution"] == [320, 180]


def test_table_default_wrist_resolution_is_used_without_a_wrist_override():
    with pytest.raises(ValueError, match="wrist camera aspect ratio"):
        table_config({"scene_overrides": {"wrist_camera_resolution": [640, 360]}})


@pytest.mark.parametrize("load", [table_config, playground_config])
def test_reset_rendering_inherits_scene_warmup_with_explicit_override_precedence(load, tmp_path):
    import json
    cfg = load({})
    assert cfg["reset_render_frames"] == cfg["scene"]["rendering"]["warmup_frames"]
    assert load({"scene_overrides": {"rendering": {"warmup_frames": 13}}})["reset_render_frames"] == 13
    override = load({"reset_render_frames": 3, "scene_overrides": {"rendering": {"warmup_frames": 13}}})
    assert override["reset_render_frames"] == 3
    assert override["scene"]["rendering"]["warmup_frames"] == 13
    # Older scenes without rendering hints retain the documented eight frames.
    scene = deepcopy(cfg["scene"])
    scene.pop("rendering", None)
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(scene))
    assert load({"scene_config": str(path)})["reset_render_frames"] == 8
