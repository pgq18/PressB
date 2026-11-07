"""Mount sweeps must move the robot without moving its room or recorded plan."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("adjust_playground_mount", ROOT/"scripts/adjust_playground_mount.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def config():
    # Historical baseline is independent of the eventual adopted default height.
    cfg = json.loads((ROOT/"configs/calibration/playground_20261107_before_mount_depth.json").read_text())
    assert cfg["robot_base_height_above_table_m"] == .014
    cfg["objects"] = copy.deepcopy(cfg["calibration_objects"])
    return cfg


@pytest.mark.parametrize("height", [.014, .010, .006])
def test_authored_room_is_fixed_but_robot_and_wrist_move(height):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom
    from pressb.tasks.playground.scene import build_scene
    from pressb.sensors.camera import make_camera, camera_intrinsics_from_usd
    from pressb.robots.piper.kinematics import PiperKinematics
    before = config()
    snapshot = copy.deepcopy(before)
    after = MODULE.adjust_mount_height(before, height)
    stages = [Usd.Stage.CreateInMemory(), Usd.Stage.CreateInMemory()]
    scenes = [build_scene(stage, cfg) for stage, cfg in zip(stages, [before, after])]
    bases = [np.array(scene["robot_base_position_world"]) for scene in scenes]
    np.testing.assert_allclose(bases[1]-bases[0], [0, 0, height-.014], atol=1e-15)
    assert before == snapshot
    assert before["objects"] == after["objects"]
    assert before["calibration_objects"] == after["calibration_objects"]
    assert before["initial_state_joint_rad_m"] == after["initial_state_joint_rad_m"]
    assert before["wrist_camera"] == after["wrist_camera"]
    assert scenes[0]["table_top_z"] == scenes[1]["table_top_z"]
    assert scenes[0]["objects"] == scenes[1]["objects"]
    for name in scenes[0]["static_props"]:
        np.testing.assert_allclose(scenes[0]["static_props"][name]["center_world"],
                                   scenes[1]["static_props"][name]["center_world"], atol=1e-15)
    for old, new in zip(scenes[0]["appearance"]["lighting"]["area_lights"],
                        scenes[1]["appearance"]["lighting"]["area_lights"]):
        for key in ("position_world_m", "target_world_m", "optical_forward_world"):
            np.testing.assert_allclose(old[key], new[key], atol=1e-15)
    for name in ("up_camera", "overview_camera"):
        poses, intrinsics = [], []
        for stage, cfg, base in zip(stages, [before, after], bases):
            path = make_camera(stage, "/"+name, cfg[name], base)
            prim = stage.GetPrimAtPath(path)
            poses.append(np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim)))
            intrinsics.append(camera_intrinsics_from_usd(prim, cfg[name]["resolution"]))
        np.testing.assert_allclose(poses[0], poses[1], atol=1e-14)
        np.testing.assert_allclose(intrinsics[0], intrinsics[1], atol=1e-14)
        for key in ("eye_base_m", "target_base_m"):
            np.testing.assert_allclose(np.array(before[name][key])+bases[0],
                                       np.array(after[name][key])+bases[1], atol=1e-15)
    # Independent URDF FK proves the same q changes TCP and wrist world Z only.
    kin = PiperKinematics(ROOT/before["robot_urdf"], tip_offset=0.)
    link6 = kin.fk(np.array(before["initial_state_joint_rad_m"][:6]))
    wrist_base = link6 @ np.array(before["wrist_camera"]["link6_from_camera_usd"])
    np.testing.assert_allclose((wrist_base[:3, 3]+bases[1])-(wrist_base[:3, 3]+bases[0]),
                               [0, 0, height-.014], atol=1e-15)


def test_control_is_identical_and_roundtrip_does_not_accumulate_motion():
    before = config()
    control = MODULE.adjust_mount_height(before, .014)
    assert control == before and control is not before
    recovered = MODULE.adjust_mount_height(MODULE.adjust_mount_height(before, .006), .014)
    recovered["reconstruction"] = before["reconstruction"]
    for name in ("up_camera", "overview_camera"):
        for key in ("eye_base_m", "target_base_m"):
            np.testing.assert_allclose(recovered[name][key], before[name][key], atol=1e-15)
    assert recovered["wrist_camera"] == before["wrist_camera"]
    assert recovered["objects"] == before["objects"]


@pytest.mark.parametrize("height", [-.001, float("nan"), float("inf")])
def test_invalid_heights_are_rejected(height):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        MODULE.adjust_mount_height(config(), height)


def test_matrix_translation_cannot_silently_drift_from_camera():
    cfg = config()
    cfg["up_camera"]["base_from_camera_usd"][3] = [0, 0, 1, 1]
    with pytest.raises(ValueError, match="homogeneous"):
        MODULE.adjust_mount_height(cfg, .006)


def test_cli_does_not_overwrite_source_or_existing_artifact(tmp_path):
    path = tmp_path/"config.json"
    path.write_text(json.dumps(config()))
    original = path.read_bytes()
    result = subprocess.run([sys.executable, str(ROOT/"scripts/adjust_playground_mount.py"),
                             "--config", str(path), "--height-m", ".006", "--output", str(path)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "FileExistsError" in result.stderr
    assert path.read_bytes() == original
