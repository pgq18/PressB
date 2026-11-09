"""CPU contracts for the observed real workbench's independent playground.

Telemetry deliberately differs from commanded joints: observations must remain
measured, and simply moving a robot must never manufacture a task success.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from pressb.runtime.contract import decode_image, validate_metadata
from pressb.runtime.simulation import SimulationService
from pressb.tasks.playground.backend import PlaygroundBackend, load_config


ROOT = Path(__file__).resolve().parents[1]
INITIAL = [0., .44, -.44, -.08, .70, .09, .02]


@pytest.fixture
def scene_config(tmp_path):
    cfg = json.loads((ROOT / "configs/piper_pick_place.json").read_text())
    cfg.update(root_path="/World/Playground", task_props=False, objects=[],
               static_props=[], tape_markers=[], task_instruction="Explore the workbench.",
               initial_state_joint_rad_m=INITIAL)
    for name in ("initial_joint_positions_rad", "initial_gripper_width_m"):
        cfg.pop(name, None)
    path = tmp_path / "playground.json"
    path.write_text(json.dumps(cfg))
    return path


class MeasuredEnvironment:
    def __init__(self, config):
        self.config = config
        self.total_ticks = 0
        self.reset_count = 0
        self.closed = False

    def reset(self, spec):
        self.spec = deepcopy(spec)
        self.reset_count += 1
        self.measured = np.asarray(spec["initial_state_joint_rad_m"], dtype=float).copy()
        self.measured[0] += .001
        self.sent = []

    def state(self):
        return self.measured.copy()

    def images(self):
        return {name: np.zeros((3, 4, 3), dtype=np.uint8) for name in ("up", "wrist")}

    def apply_action(self, command):
        self.sent.append(np.asarray(command).copy())

    def tick(self):
        self.total_ticks += 1
        return {}

    def close(self):
        self.closed = True


def backend(scene_config, **overrides):
    return PlaygroundBackend({"scene_config": str(scene_config), "settling_ticks": 0,
        "max_episode_seconds": 2 / 30, **overrides}, environment_factory=MeasuredEnvironment)


def actions(count=1, **changes):
    command = np.asarray(INITIAL, dtype=float)
    for index, value in changes.items():
        command[int(index)] = value
    return np.tile(command, (count, 1)).tolist()


def reset(node, options=None):
    return node.reset([0], 12, {} if options is None else options)[0]


def test_metadata_and_reset_report_measured_joint7_without_starting_isaac(scene_config):
    node = backend(scene_config)
    assert node.environment is None
    metadata = validate_metadata(node.describe(), "simulation")
    assert metadata["task_id"] == "playground"
    assert metadata["num_envs"] == 1 and not metadata["execution"]["vectorized"]
    assert metadata["action"]["units"] == ["rad"] * 6 + ["m"]
    assert len(metadata["action"]["names"]) == 7
    row = reset(node)
    observation = row["observation"]
    np.testing.assert_allclose(observation["state"], node.environment.measured)
    np.testing.assert_allclose(observation["control_state"], node.environment.measured)
    assert observation["state"][0] != INITIAL[0]
    assert observation["task_features"] == []
    assert observation["task"] == "Explore the workbench."
    assert observation["model_inputs"] == {}
    assert not row["info"]["success"]
    for name in ("up", "wrist"):
        assert decode_image(observation["images"][name]).size == (4, 3)


def test_execution_keeps_units_and_measured_state_separate_from_sent_command(scene_config):
    node = backend(scene_config)
    reset(node)
    commanded = actions(**{"0": .1, "6": .045})
    row = node.step([{"env_id": 0, "actions": commanded}])[0]
    np.testing.assert_allclose(node.environment.sent[0], commanded[0])
    np.testing.assert_allclose(row["observation"]["control_state"], commanded[0])
    np.testing.assert_allclose(row["observation"]["state"], node.environment.measured)
    assert row["reward"] == 0. and not row["terminated"] and not row["truncated"]
    assert not row["info"]["success"]
    assert row["info"]["physics_steps"] == 4
    assert row["discount"] == pytest.approx(.99)


def test_timeout_counts_only_executed_chunk_prefix_and_requires_explicit_reset(scene_config):
    node = backend(scene_config)
    reset(node)
    row = node.step([{"env_id": 0, "actions": actions(7)}])[0]
    assert row["truncated"] and not row["terminated"]
    assert row["reward"] == 0 and not row["info"]["success"]
    assert row["info"]["requested_action_count"] == 7
    assert row["info"]["elapsed_control_steps"] == 2
    assert row["info"]["physics_steps"] == 8
    assert len(row["info"]["applied_actions_rad_m"]) == 2
    assert row["discount"] == pytest.approx(.99**2)
    with pytest.raises(ValueError, match="[Rr]eset"):
        node.step([{"env_id": 0, "actions": actions()}])
    reset(node)
    assert not node.step([{"env_id": 0, "actions": actions()}])[0]["truncated"]


def test_single_interval_duration_is_not_rounded_up_or_down(scene_config):
    node = backend(scene_config, max_episode_seconds=1 / 30)
    reset(node)
    row = node.step([{"env_id": 0, "actions": actions(7)}])[0]
    assert row["truncated"] and row["info"]["elapsed_control_steps"] == 1
    assert row["info"]["physics_steps"] == 4
    assert len(node.environment.sent) == 1


def test_reset_overrides_pose_task_and_model_inputs_without_mutating_config(scene_config):
    inputs = {"camera_order": ["wrist", "up"]}
    node = backend(scene_config, model_inputs=inputs)
    alternative = list(INITIAL)
    alternative[0], alternative[6] = .12, .04
    options = {"initial_state_joint_rad_m": alternative, "task": "Observe the blue cube."}
    original = deepcopy(options)
    row = reset(node, options)
    assert options == original
    assert row["observation"]["task"] == original["task"]
    assert row["observation"]["model_inputs"] == inputs
    np.testing.assert_allclose(node.environment.spec["initial_state_joint_rad_m"], alternative)
    reset(node)
    np.testing.assert_allclose(node.environment.spec["initial_state_joint_rad_m"], INITIAL)


def test_smoothing_retains_history_across_chunks_without_filtering_gripper(scene_config):
    node = backend(scene_config, max_episode_seconds=3 / 30,
                   smoothing={"mode": "joint_ema", "alpha": .25})
    initial = np.asarray(reset(node)["observation"]["state"])
    command = np.asarray(actions(**{"0": .4, "6": .05})[0])
    node.step([{"env_id": 0, "actions": [command.tolist()]}])
    node.step([{"env_id": 0, "actions": [command.tolist()]}])
    first, second = node.environment.sent
    np.testing.assert_allclose(first[:6], .25 * command[:6] + .75 * initial[:6])
    np.testing.assert_allclose(second[:6], .25 * command[:6] + .75 * first[:6])
    assert first[6] == pytest.approx(.05) and second[6] == pytest.approx(.05)


def test_service_retries_do_not_repeat_reset_or_advance_physics(scene_config):
    node = backend(scene_config)
    service = SimulationService(node)
    request = {"run_id": "playground-test", "request_id": "reset", "env_ids": [0],
               "seed": 4, "options": {}}
    initial = service.reset(request)["items"][0]
    assert service.reset(request)["items"][0] == initial
    assert node.environment.reset_count == 1
    step = {"run_id": "playground-test", "request_id": "step", "items": [
        {"env_id": 0, "episode_id": initial["episode_id"], "step_id": 0, "actions": actions(7)}]}
    result = service.step(step)
    assert service.step(step) == result
    assert node.environment.total_ticks == 8
    service.close()
    assert node.environment.closed


@pytest.mark.parametrize("command", [
    [100., .44, -.44, -.08, .70, .09, .02],
    [0., .44, -.44, -.08, .70, .09, -.01],
    [0., .44, -.44, -.08, .70, .09, float("nan")],
    INITIAL[:6],
])
def test_invalid_action_does_not_advance_or_poison_backend(scene_config, command):
    node = backend(scene_config)
    reset(node)
    with pytest.raises(ValueError):
        node.step([{"env_id": 0, "actions": [command]}])
    assert node.environment.total_ticks == 0 and node.environment.sent == []
    node.step([{"env_id": 0, "actions": actions()}])


@pytest.mark.parametrize("options", [
    {"target_color": "red"}, {"initial_state_joint_rad_m": [0.] * 6},
    {"initial_state_joint_rad_m": [100.] + INITIAL[1:]},
    {"initial_state_joint_rad_m": INITIAL[:6] + [-.01]},
    {"initial_state_joint_rad_m": INITIAL[:6] + [float("nan")]},
])
def test_invalid_reset_options_rejected_before_environment_creation(scene_config, options):
    node = backend(scene_config)
    with pytest.raises(ValueError):
        reset(node, options)
    assert node.environment is None


@pytest.mark.parametrize("changes", [
    {"num_envs": 0}, {"action_dim": 6}, {"physics_dt": .01},
    {"max_episode_seconds": .05}, {"gamma": 1.1}, {"gamma": 0.},
    {"headless": "true"}, {"settling_ticks": -1},
    {"scene_overrides": {"initial_state_joint_rad_m": [0.] * 6}},
    {"scene_overrides": {"initial_state_joint_rad_m": INITIAL[:6] + [float("nan")]}},
])
def test_invalid_config_fails_before_launching_isaac(scene_config, changes):
    with pytest.raises(ValueError):
        load_config({"scene_config": str(scene_config), **changes})


def test_physics_failure_latches_fault_and_close_still_releases_environment(scene_config, monkeypatch):
    node = backend(scene_config)
    reset(node)
    def failed_tick():
        raise RuntimeError("physics failed after command")
    monkeypatch.setattr(node.environment, "tick", failed_tick)
    with pytest.raises(RuntimeError, match="physics failed"):
        node.step([{"env_id": 0, "actions": actions()}])
    with pytest.raises(RuntimeError, match="faulted"):
        node.step([{"env_id": 0, "actions": actions()}])
    with pytest.raises(RuntimeError, match="faulted"):
        reset(node)
    node.close()
    assert node.environment.closed


def test_identity_tracks_scene_calibration_but_not_execution_gpu(scene_config):
    first = backend(scene_config)
    assert first.identity == backend(scene_config, gpu=1).identity
    changed = backend(scene_config, scene_overrides={"up_camera": {"fx": 600.}})
    assert first.identity["task_config_sha256"] != changed.identity["task_config_sha256"]


def test_empty_playground_has_workbench_without_old_task_props_or_rigid_objects(scene_config):
    Usd = pytest.importorskip("pxr.Usd")
    from pxr import UsdPhysics
    from pressb.tasks.playground.scene import build_scene
    cfg = json.loads(scene_config.read_text())
    original = deepcopy(cfg)
    stage = Usd.Stage.CreateInMemory()
    scene = build_scene(stage, cfg)
    assert cfg == original
    root = scene["root_path"]
    assert root == "/World/Playground"
    assert stage.GetPrimAtPath(root + "/Table/Top")
    assert scene["objects"] == {} and not scene["task_props_enabled"]
    for name in ("Box", "Tape", "Object", "Cube", "Buttons"):
        assert not stage.GetPrimAtPath(root + "/" + name)
    assert not [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]


def test_optional_object_is_dynamic_but_background_props_default_to_visual_only(scene_config):
    Usd = pytest.importorskip("pxr.Usd")
    from pxr import UsdPhysics
    from pressb.tasks.playground.scene import build_scene
    cfg = json.loads(scene_config.read_text())
    cfg["objects"] = [{"name": "BlueCube", "center_xy": [.3, .1],
                       "size": [.03] * 3, "mass": .025, "color": [.02, .03, .8]}]
    cfg["static_props"] = [{"name": "Equipment", "center_base_m": [.7, .3, .1],
                            "size": [.2, .1, .2], "color": [.1, .1, .1]}]
    stage = Usd.Stage.CreateInMemory()
    scene = build_scene(stage, cfg)
    cube = scene["objects"]["BlueCube"]
    prop = scene["static_props"]["Equipment"]
    cube_prim = stage.GetPrimAtPath(cube["path"])
    prop_prim = stage.GetPrimAtPath(prop["path"])
    assert cube_prim.HasAPI(UsdPhysics.RigidBodyAPI)
    assert cube_prim.HasAPI(UsdPhysics.CollisionAPI)
    assert not UsdPhysics.RigidBodyAPI(cube_prim).GetKinematicEnabledAttr().Get()
    assert UsdPhysics.MassAPI(cube_prim).GetMassAttr().Get() == pytest.approx(.025)
    assert cube["center_world"][2] == pytest.approx(scene["table_top_z"] + .0155)
    assert cube["path"] in scene["collision_paths"]
    assert not prop_prim.HasAPI(UsdPhysics.RigidBodyAPI)
    assert not prop_prim.HasAPI(UsdPhysics.CollisionAPI)
    assert prop["path"] not in scene["collision_paths"]
    assert prop["collision"] is False


def test_rotated_object_footprint_must_fit_table_even_if_its_center_does(scene_config):
    Usd = pytest.importorskip("pxr.Usd")
    from pressb.tasks.playground.scene import build_scene
    cfg = json.loads(scene_config.read_text())
    # At 45 degrees, a square half-extent grows from 15 mm to 21.2 mm.
    cfg["objects"] = [{"name": "Cube", "center_xy": [cfg["table_x_min"] + .02, 0.],
                       "size": [.03] * 3, "yaw_deg": 45.}]
    with pytest.raises(ValueError, match="supported"):
        build_scene(Usd.Stage.CreateInMemory(), cfg)


def test_base_mount_height_separates_robot_frame_from_dynamic_table_objects(scene_config):
    Usd = pytest.importorskip("pxr.Usd")
    from pxr import UsdGeom, UsdPhysics
    from pressb.tasks.playground.scene import build_scene
    cfg = json.loads(scene_config.read_text())
    cfg.update(robot_base_height_above_table_m=.014, mount_hardware=True,
               objects=[{"name": "Cube", "center_xy": [.3, .1], "size": [.029]*3}],
               static_props=[{"name": "Prop", "center_base_m": [.7, .3, .1], "size": [.1]*3}])
    stage = Usd.Stage.CreateInMemory()
    scene = build_scene(stage, cfg)
    table_z = scene["table_top_z"]
    assert scene["robot_base_position_world"][2] == pytest.approx(table_z + .014)
    assert scene["objects"]["Cube"]["center_world"][2] == pytest.approx(table_z + .0145 + .0005)
    assert scene["static_props"]["Prop"]["center_world"][2] == pytest.approx(table_z + .014 + .1)
    mount = scene["mount_hardware"]
    prim = stage.GetPrimAtPath(mount["root_path"] + "/BasePlate")
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(prim).ComputeAlignedRange()
    assert bbox.GetMin()[2] == pytest.approx(table_z)
    assert bbox.GetMax()[2] == pytest.approx(scene["robot_base_position_world"][2])
    assert not prim.HasAPI(UsdPhysics.CollisionAPI)


def test_zero_mount_height_preserves_legacy_scene_exactly(scene_config):
    Usd = pytest.importorskip("pxr.Usd")
    from pressb.tasks.playground.scene import build_scene
    cfg = json.loads(scene_config.read_text())
    first, second = Usd.Stage.CreateInMemory(), Usd.Stage.CreateInMemory()
    legacy = build_scene(first, cfg)
    explicit_zero = build_scene(second, {**cfg, "robot_base_height_above_table_m": 0.})
    assert legacy == explicit_zero
    assert first.GetRootLayer().ExportToString() == second.GetRootLayer().ExportToString()


@pytest.mark.parametrize("height", [-.001, float("nan"), float("inf")])
def test_invalid_mount_height_rejected_before_scene_authoring(scene_config, height):
    Usd = pytest.importorskip("pxr.Usd")
    from pressb.tasks.playground.scene import build_scene
    cfg = json.loads(scene_config.read_text())
    stage = Usd.Stage.CreateInMemory()
    with pytest.raises(ValueError, match="robot_base_height_above_table_m"):
        build_scene(stage, {**cfg, "robot_base_height_above_table_m": height})
    assert not stage.GetPrimAtPath("/World/Playground")


def test_mount_migration_preserves_static_props_and_area_light_world_coordinates(scene_config):
    Usd = pytest.importorskip("pxr.Usd")
    from pressb.tasks.playground.scene import build_scene
    cfg = json.loads(scene_config.read_text())
    cfg["lighting"] = {"area_lights": [{"position_base_m": [.3, .2, 1.],
                                        "target_base_m": [.3, .2, 0.], "width": .5, "height": .5}]}
    cfg["static_props"] = [{"name": "Prop", "center_base_m": [.7, .3, .1], "size": [.1]*3}]
    before = build_scene(Usd.Stage.CreateInMemory(), cfg)
    migrated = deepcopy(cfg)
    migrated["robot_base_height_above_table_m"] = .014
    migrated["static_props"][0]["center_base_m"][2] -= .014
    for key in ("position_base_m", "target_base_m"):
        migrated["lighting"]["area_lights"][0][key][2] -= .014
    after = build_scene(Usd.Stage.CreateInMemory(), migrated)
    assert before["static_props"]["Prop"]["center_world"] == pytest.approx(after["static_props"]["Prop"]["center_world"])
    old_light, new_light = (scene["appearance"]["lighting"]["area_lights"][0] for scene in (before, after))
    for key in ("position_world_m", "target_world_m"):
        assert old_light[key] == pytest.approx(new_light[key])
