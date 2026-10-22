"""Real-asset collision checks without Isaac or a GPU."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pin = pytest.importorskip("pinocchio")
pytest.importorskip("hppfcl")

from pressb.kinematics import PiperKinematics
from pressb.table_buttons.collision import RobotClearance


ROOT = Path(__file__).resolve().parents[1]
TABLE_COLLISION = np.array([.07633662492192972, 2.174618359685737, -.12898172555869136,
                            -.5866554866909506, -.9988746914093674, -2.073257203013838])
SELF_COLLISION = np.array([-1.372192027619624, 2.652725280532363, -.011321859844994453,
                           -.5954437396947299, 1.1262145962028354, .49942884138190413])


@pytest.fixture(scope="module")
def scene():
    cfg = json.loads((ROOT / "configs/piper_table_buttons.json").read_text())
    path = ROOT / cfg["robot_urdf"]
    if not path.is_file():
        pytest.skip("Downloaded Piper asset is required")
    kin = PiperKinematics(path, tip_offset=.1358)
    checker = RobotClearance(kin, cfg)
    return cfg, kin, checker


def press_pose(scene, z=.025, pitch=160.):
    cfg, kin, _ = scene
    xy = np.array([.33, .12])
    rotation = (Rotation.from_euler("z", np.arctan2(xy[1], xy[0])).as_matrix()
                @ Rotation.from_euler("y", pitch, degrees=True).as_matrix())
    return kin.solve(np.r_[xy, z], np.array(cfg["initial_joint_positions_rad"]), rotation)


@pytest.mark.parametrize("width", [.0002, .06, .1])
def test_finger_and_mesh_frames_match_independent_pinocchio_fk(scene, width):
    cfg, kin, checker = scene
    model = pin.buildModelFromUrdf(str(kin.urdf_path))
    geom = pin.buildGeomFromUrdf(model, str(kin.urdf_path), pin.GeometryType.COLLISION,
                                package_dirs=[str(kin.urdf_path.parent)])
    data, gdata = model.createData(), geom.createData()
    q = np.array(cfg["initial_joint_positions_rad"])
    pin.updateGeometryPlacements(model, data, geom, gdata, np.r_[q, width/2, -width/2])
    frames = checker.link_transforms(q, width)
    for index, item in enumerate(checker.geometries[:10]):
        np.testing.assert_allclose(frames[item.link] @ item.local_pose,
                                   gdata.oMg[index].homogeneous, atol=2e-14, rtol=0)
    assert checker.check(q, width)[0]


def test_wrist_uses_physical_housing_registration_not_optical_frame(scene):
    _, _, checker = scene
    camera = next(item for item in checker.geometries if item.name == "wrist_camera")
    stand = next(item for item in checker.geometries if item.name == "wrist_stand")
    # Independent reference from the exported physical scene's link6-relative
    # colliders; PAP's optical registration deliberately differs from these.
    np.testing.assert_allclose(camera.local_pose[:3, 3],
                               [-.0730960945, -.0290064471, .0322500250], atol=1e-10)
    np.testing.assert_allclose(stand.local_pose[:3, 3], [-.032, -.001, .018], atol=1e-12)


def test_known_real_mesh_table_penetration_and_finite_table_footprint(scene):
    cfg, kin, checker = scene
    clear, diagnostic = checker.check(TABLE_COLLISION, .0002)
    assert not clear and diagnostic["reason"] == "table_collision"
    assert diagnostic["pair"] == ["gripper_base_0", "table"]
    assert diagnostic["penetration_depth_m"] > .001
    # Same below-Z=0 robot pose, but the finite slab is now beyond its reach.
    moved_table = RobotClearance(kin, {**cfg, "table_x_min": 2.})
    assert moved_table.check(TABLE_COLLISION, .0002)[0]


def test_nonadjacent_link_base_collision_is_not_filtered(scene):
    _, _, checker = scene
    clear, diagnostic = checker.check(SELF_COLLISION, .0002)
    assert not clear and diagnostic["reason"] == "self_collision"
    assert diagnostic["pair"] == ["link4_0", "base_link_0"]


def test_closed_finger_siblings_are_not_silently_excluded(scene):
    cfg, _, checker = scene
    # The source URDF's rounded rotations produce a tiny true mesh overlap at
    # exactly zero width. The task's actual 0.2 mm initial opening is clear.
    clear, diagnostic = checker.check(cfg["initial_joint_positions_rad"], 0.)
    assert not clear and diagnostic["reason"] == "self_collision"
    assert set(diagnostic["pair"]) == {"link7_0", "link8_0"}
    assert checker.check(cfg["initial_joint_positions_rad"], cfg["initial_gripper_width_m"])[0]


def test_path_reports_first_unsafe_sample_and_honors_per_sample_width(scene):
    cfg, _, checker = scene
    initial = cfg["initial_joint_positions_rad"]
    clear, diagnostic = checker.check_path([initial, TABLE_COLLISION], [.06, .0002])
    assert not clear and diagnostic["sample_index"] == 1
    assert diagnostic["checked_samples"] == 2 and diagnostic["reason"] == "table_collision"


@pytest.mark.parametrize("pitch", [140., 150., 160.])
def test_reachable_press_orientations_use_same_collision_model(scene, pitch):
    cfg, _, checker = scene
    pose = press_pose(scene, z=.04, pitch=pitch)
    assert checker.check(pose, cfg["initial_gripper_width_m"])[0]


def test_only_target_cap_is_allowed_to_contact_fingers(scene):
    cfg, _, checker = scene
    pose = press_pose(scene)
    positions = {"red": [.33, .12], "blue": [.33, -.12]}
    width = cfg["initial_gripper_width_m"]
    assert checker.check_button_path([pose], positions, "red", width)[0]
    clear, diagnostic = checker.check_button_path([pose], positions, "blue", width)
    assert not clear and diagnostic["reason"] == "button_collision"
    assert diagnostic["pair"][1] == "red/cap"


def test_target_housing_and_bezel_remain_collision_obstacles(scene):
    cfg, _, checker = scene
    width = cfg["initial_gripper_width_m"]
    clear, diagnostic = checker.check_button_path([press_pose(scene, z=.01)],
                                                  {"red": [.33, .12]}, "red", width)
    assert not clear and diagnostic["pair"][1] == "red/housing"
    clear, diagnostic = checker.check_button_path([press_pose(scene, z=.015)],
                                                  {"red": [.34, .12]}, "red", width)
    assert not clear and diagnostic["pair"][1].startswith("red/bezel_")


@pytest.mark.parametrize("width", [-.001, .1001, float("nan")])
def test_gripper_width_outside_explicit_overlay_is_rejected(scene, width):
    cfg, _, checker = scene
    with pytest.raises(ValueError, match="gripper width"):
        checker.check(cfg["initial_joint_positions_rad"], width)


def test_missing_target_and_empty_path_do_not_pass_screening(scene):
    cfg, _, checker = scene
    with pytest.raises(ValueError, match="target"):
        checker.check_button_path([cfg["initial_joint_positions_rad"]], {"red": [.33, .12]}, "blue", .0002)
    with pytest.raises(ValueError, match="nonempty"):
        checker.check_path(np.empty((0, 6)), .0002)
