"""Language selection, physical switch evidence, and preserved PAP workbench."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pressb.table_buttons.task import (
    COLORS, TableButtonOutcome, button_in_camera, parse_instruction,
    project_base_points, sample_button_positions, task_instruction,
)


ROOT = Path(__file__).resolve().parents[1]


def config(name="piper_table_buttons.json"):
    return json.loads((ROOT / "configs" / name).read_text())


def sample(checker, seconds, *, depths=None, forces=None):
    return checker.update(seconds,
                          {c: float((depths or {}).get(c, 0.)) for c in COLORS},
                          {c: float((forces or {}).get(c, 0.)) for c in COLORS})


@pytest.mark.parametrize("color", COLORS)
def test_canonical_instructions_select_exactly_the_named_color(color):
    assert parse_instruction(task_instruction(color)) == color
    assert parse_instruction(f"  PLEASE   PRESS {color.upper()} BUTTON!  ") == color


@pytest.mark.parametrize("text,color", [
    ("按红色按钮", "red"), ("请按下蓝色按键。", "blue"), ("按下黄按钮", "yellow"),
])
def test_chinese_single_color_instructions(text, color):
    assert parse_instruction(text) == color


@pytest.mark.parametrize("text", [
    "Do not press the red button.", "Press the red and blue buttons.",
    "不要按红色按钮", "按红色按钮然后按蓝色按钮", "Press the green button.", "red", "",
])
def test_negated_multitarget_or_unknown_instructions_are_not_silently_accepted(text):
    with pytest.raises(ValueError):
        parse_instruction(text)


@pytest.mark.parametrize("depth,force", [(0., 5.), (.0029, 2.), (.004, 0.), (.004, .099)])
def test_force_and_cap_travel_are_both_necessary(depth, force):
    checker = TableButtonOutcome("red")
    sample(checker, 0., depths={"red": depth}, forces={"red": force})
    result = sample(checker, 1., depths={"red": depth}, forces={"red": force})
    assert not result["success"] and not result["terminated"] and not result["press_events"]


def test_sustained_measured_press_succeeds_after_dwell_and_latches_time():
    checker = TableButtonOutcome("blue", hold_seconds=.025)
    assert not sample(checker, .1, depths={"blue": .0032}, forces={"blue": .7})["success"]
    assert not sample(checker, .124, depths={"blue": .0032}, forces={"blue": .7})["success"]
    result = sample(checker, .125, depths={"blue": .0032}, forces={"blue": .7})
    assert result["success"] and result["terminated"] and result["reward"] == 1.
    assert result["first_success_time_s"] == .125
    assert result["press_events"] == ["blue"]
    held = sample(checker, .5, depths={"blue": .004}, forces={"blue": 2.})
    assert held["first_success_time_s"] == .125 and held["press_events"] == []


@pytest.mark.parametrize("interruption", [dict(depths={"red": .002}), dict(forces={"red": 0.})])
def test_contact_loss_or_shallow_travel_restarts_dwell(interruption):
    checker = TableButtonOutcome("red")
    valid = dict(depths={"red": .004}, forces={"red": 1.})
    sample(checker, 0., **valid)
    sample(checker, .02, **{**valid, **interruption})
    assert not sample(checker, .03, **valid)["success"]
    assert not sample(checker, .05, **valid)["success"]
    assert sample(checker, .055, **valid)["success"]


def test_wrong_color_is_terminal_and_later_target_cannot_recover_the_episode():
    checker = TableButtonOutcome("yellow")
    sample(checker, 0., depths={"red": .004}, forces={"red": 1.})
    wrong = sample(checker, .03, depths={"red": .004}, forces={"red": 1.})
    assert wrong["wrong_button"] == "red" and wrong["terminated"]
    assert wrong["reward"] == 0. and not wrong["success"]
    sample(checker, .1, depths={"yellow": .004}, forces={"yellow": 1.})
    result = sample(checker, .2, depths={"yellow": .004}, forces={"yellow": 1.})
    assert result["wrong_button"] == "red" and not result["success"]
    assert result["failure_time_s"] == .03


def test_simultaneous_target_and_wrong_button_is_not_success():
    checker = TableButtonOutcome("red", hold_seconds=0.)
    result = sample(checker, 0., depths={"red": .004, "blue": .004}, forces={"red": 1., "blue": 1.})
    assert result["wrong_button"] == "blue" and not result["success"]


def test_incidental_contact_on_other_buttons_does_not_fail_target_press():
    checker = TableButtonOutcome("yellow", hold_seconds=0.)
    result = sample(checker, 0., depths={"red": .001, "yellow": .004}, forces={"red": 2., "yellow": 2.})
    assert result["success"] and result["wrong_button"] is None


def test_spring_release_hysteresis_prevents_repeated_events_until_full_release():
    checker = TableButtonOutcome("red", hold_seconds=0.)
    assert sample(checker, 0., depths={"red": .004}, forces={"red": 1.})["press_events"] == ["red"]
    sample(checker, .1, depths={"red": .002})
    assert sample(checker, .2, depths={"red": .004}, forces={"red": 1.})["press_events"] == []
    sample(checker, .3, depths={"red": .0009})
    assert sample(checker, .4, depths={"red": .004}, forces={"red": 1.})["press_events"] == ["red"]


def test_incomplete_contact_observation_cannot_hide_a_wrong_press():
    checker = TableButtonOutcome("red")
    with pytest.raises(ValueError, match="all three"):
        checker.update(0., {"red": .004}, {"red": 1.})


def test_projection_uses_usd_negative_z_and_image_down_v():
    camera = dict(base_from_camera_usd=np.eye(4), resolution=[640, 480], fx=500., fy=500., cx=320., cy=240.)
    pixels, depth = project_base_points([[0., 0., -1.], [.1, .1, -1.]], camera)
    np.testing.assert_allclose(pixels, [[320., 240.], [370., 190.]])
    np.testing.assert_allclose(depth, [1., 1.])


def test_default_buttons_are_separated_and_fully_inside_inherited_camera():
    cfg = config()
    points = []
    for color in COLORS:
        xy = cfg["buttons"][color]["center_xy"]
        assert button_in_camera(xy, cfg["up_camera"], margin_px=16.)
        points.append(xy)
    distances = np.linalg.norm(np.asarray(points)[:, None] - np.asarray(points)[None], axis=-1)
    assert np.min(distances + np.eye(3)) >= .10


def test_random_positions_are_reproducible_separated_and_visible():
    camera = config()["up_camera"]
    assert sample_button_positions(camera, 17) == sample_button_positions(camera, 17)
    assert sample_button_positions(camera, 17) != sample_button_positions(camera, 18)
    for seed in range(25):
        positions = sample_button_positions(camera, seed, margin_px=16.)
        assert set(positions) == set(COLORS)
        for color, xy in positions.items():
            assert button_in_camera(xy, camera, margin_px=16.)
            assert .30 <= xy[0] <= .39 and -.14 <= xy[1] <= .14
            for other in COLORS:
                if color != other:
                    assert np.linalg.norm(np.asarray(xy) - positions[other]) >= .10


def test_impossible_sampling_bounds_raise_without_reusing_overlapping_positions():
    with pytest.raises(ValueError, match="Cannot place"):
        sample_button_positions(config()["up_camera"], 0, bounds_xy=((.3, .31), (0., .01)), max_attempts=100)


def test_config_keeps_pap_workbench_and_fixed_cameras_with_default_wrist_optics():
    cfg, pap = config(), config("piper_pick_place_episode_461.json")
    for key in ("robot_usd", "robot_urdf", "robot_base_xy", "table_height", "table_depth", "table_width",
                "table_x_min", "up_camera", "overview_camera", "lighting", "rendering"):
        assert cfg[key] == pap[key]
    assert "wrist_camera" not in cfg  # Same attach_wrist_camera defaults as elevator.
    assert cfg["wrist_camera_resolution"] == [640, 480]
    assert cfg["materials"]["Mat"] == pap["materials"]["Mat"]
    assert cfg["initial_pose_source"]["episode_index"] == 461
    assert cfg["task_props"] is False and cfg["tape_markers"] == []
    assert "box" not in cfg and "cube" not in cfg and "object" not in cfg


def test_default_and_workspace_corner_approaches_have_bounded_ik():
    from pressb.kinematics import PiperKinematics
    cfg = config()
    kin = PiperKinematics(ROOT / cfg["robot_urdf"], tip_offset=.1358)
    seed = cfg["initial_joint_positions_rad"]
    points = [value["center_xy"] for value in cfg["buttons"].values()]
    points += [[x, y] for x in (.30, .39) for y in (-.14, .14)]
    for x, y in points:
        rotation = Rotation.from_euler("z", np.arctan2(y, x)).as_matrix() @ Rotation.from_euler("y", 160., degrees=True).as_matrix()
        for z in (.12, .02):
            q = kin.solve([x, y, z], seed, rotation)
            np.testing.assert_allclose(kin.fk(q)[:3, 3], [x, y, z], atol=5e-4)
            assert np.all(q >= kin.lower) and np.all(q <= kin.upper)


def test_scene_has_only_three_dynamic_caps_and_preserves_original_pap_props():
    Usd = pytest.importorskip("pxr.Usd")
    from pxr import UsdPhysics
    from pressb.table_buttons.scene import build_scene
    from pressb.pick_place.scene import build_scene as build_pap
    cfg = config()
    original = deepcopy(cfg)
    stage = Usd.Stage.CreateInMemory()
    result = build_scene(stage, cfg)
    assert cfg == original
    root = result["root_path"]
    for name in ("Tape", "Box", "Object", "Cube"):
        assert not stage.GetPrimAtPath(f"{root}/{name}")
    bodies = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert {str(prim.GetPath()) for prim in bodies} == {value["body_path"] for value in result["buttons"].values()}
    for color, metadata in result["buttons"].items():
        joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(metadata["joint_path"]))
        assert joint.GetAxisAttr().Get() == "Z"
        assert joint.GetLowerLimitAttr().Get() == pytest.approx(-.004)
        assert joint.GetUpperLimitAttr().Get() == 0.
        assert not UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(metadata["body_path"])).GetKinematicEnabledAttr().Get()
        assert metadata["neutral_top_world"][2] - result["table_top_z"] == pytest.approx(.025)
        drive = UsdPhysics.DriveAPI(joint.GetPrim(), "linear")
        assert drive.GetStiffnessAttr().Get() > 0. and drive.GetTargetPositionAttr().Get() == 0.
    pap_stage = Usd.Stage.CreateInMemory()
    pap = build_pap(pap_stage, config("piper_pick_place_episode_461.json"))
    assert pap_stage.GetPrimAtPath(pap["box_path"])
    assert pap_stage.GetPrimAtPath(pap["object_path"])
    assert pap_stage.GetPrimAtPath(pap["root_path"] + "/Tape/Marker0")
    for suffix in ("Table/Top", "Table/FrontTrim"):
        new_prim = stage.GetPrimAtPath(root + "/" + suffix)
        old_prim = pap_stage.GetPrimAtPath(pap["root_path"] + "/" + suffix)
        assert new_prim and old_prim
        for attr in ("xformOp:translate", "xformOp:scale"):
            assert new_prim.GetAttribute(attr).Get() == old_prim.GetAttribute(attr).Get()
