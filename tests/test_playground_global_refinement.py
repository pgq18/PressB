"""Historical global-camera registration and the later opening-limit change."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "configs/calibration/playground_20261108_final.json"
NEW = ROOT / "configs/calibration/playground_20261109_global_refined.json"
CURRENT = ROOT / "configs/calibration/playground_20261113_appearance.json"
POSE_FIELDS = {"eye_base_m", "target_base_m", "base_from_camera_usd"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def runtime_without_global_pose(scene):
    scene = deepcopy(scene)
    scene.pop("reconstruction")
    for key in POSE_FIELDS | {"status"}:
        scene["up_camera"].pop(key)
    return scene


def test_previous_preset_is_immutable_and_working_alias_matches_current_version():
    assert hashlib.sha256(OLD.read_bytes()).hexdigest() == (
        "446f8eb0f264d15c37c7050e91319472b7f097d890c29ef8dc5fb3a963a34b21")
    assert (ROOT / "configs/piper_playground.json").read_bytes() == CURRENT.read_bytes()


def test_global_registration_and_subsequent_explicit_opening_limit_are_the_only_changes():
    old, new = (json.loads(path.read_text()) for path in (OLD, NEW))
    # The user subsequently widened the active 09 template to 100 mm; 08 is
    # retained at 70 mm. That independent change must not be silently ignored
    # or misrepresented as part of a camera-only historical migration.
    assert old["gripper_max_width_m"] == .07
    assert new["gripper_max_width_m"] == .1
    old_comparable = deepcopy(old)
    old_comparable["gripper_max_width_m"] = new["gripper_max_width_m"]
    # Canonical bytes also distinguish 0.0 from -0.0 and preserve exact floats.
    # This covers K/resolution, wrist/overview, home, mount, geometry, objects,
    # dynamics, materials, lights and all other runtime fields, not a shortlist.
    assert canonical(runtime_without_global_pose(old_comparable)) == canonical(runtime_without_global_pose(new))
    for key in POSE_FIELDS:
        assert canonical(old["up_camera"][key]) != canonical(new["up_camera"][key])
    for key in ("link6_from_camera_usd",):
        assert np.asarray(old["wrist_camera"][key], dtype="<f8").tobytes() == np.asarray(
            new["wrist_camera"][key], dtype="<f8").tobytes()
    assert canonical(new["reconstruction"]["final_revision"]) == canonical(old["reconstruction"]["final_revision"])


def test_global_pose_is_rigid_and_eye_target_agree_with_optical_frame():
    camera = json.loads(NEW.read_text())["up_camera"]
    transform = np.asarray(camera["base_from_camera_usd"], dtype=np.float64)
    np.testing.assert_array_equal(transform[3], [0., 0., 0., 1.])
    np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-12)
    assert np.linalg.det(transform[:3, :3]) == pytest.approx(1., abs=1e-12)
    np.testing.assert_array_equal(transform[:3, 3], camera["eye_base_m"])
    look = np.asarray(camera["target_base_m"]) - camera["eye_base_m"]
    np.testing.assert_allclose(look / np.linalg.norm(look), -transform[:3, 2], atol=1e-12)


@pytest.mark.parametrize("task", ["playground", "cube_stack", "tabletop_cleanup"])
def test_default_retains_registered_camera_while_explicit_old_template_remains_supported(task):
    import importlib
    module = "backend" if task == "playground" else "configuration"
    loader = importlib.import_module(f"pressb.tasks.{task}.{module}").load_config
    prior_options = {"scene_config": str(OLD)}
    if task != "playground":
        prior_options["initial_pose"] = {"mode": "scene"}
    current, prior = loader({}), loader(prior_options)
    assert canonical(current["scene"]["up_camera"]) == canonical(json.loads(NEW.read_text())["up_camera"])
    assert canonical(prior["scene"]["up_camera"]) == canonical(json.loads(OLD.read_text())["up_camera"])
    # Current appearance and centered home evolved after the camera fit. The
    # exact 08/09 non-camera comparison above owns the historical invariant.
    assert canonical(current["scene"]["wrist_camera"]) == canonical(prior["scene"]["wrist_camera"])


@pytest.mark.parametrize("task", ["cube_stack", "tabletop_cleanup"])
def test_camera_migration_preserves_complete_planned_commands_bitwise(task):
    import importlib
    loader = importlib.import_module(f"pressb.tasks.{task}.configuration").load_config
    planner = importlib.import_module(f"pressb.tasks.{task}.planning")
    # Preserve the recorded home for this historical comparison. The newer
    # centered-home default requests 100 mm and is unavailable in the 08 scene.
    old, new = (loader({"scene_config": str(path), "initial_pose": {"mode": "scene"}})
                for path in (OLD, NEW))
    if task == "cube_stack":
        before, after = planner.plan_stack(old["scene"]), planner.plan_stack(new["scene"])
    else:
        before, after = planner.plan_cleanup(old), planner.plan_cleanup(new)
    assert before["phases"] == after["phases"]
    assert np.asarray(before["actions"], dtype="<f8").tobytes() == np.asarray(
        after["actions"], dtype="<f8").tobytes()
