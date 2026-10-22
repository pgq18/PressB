"""Shape-aware containment contracts, independent of simulator execution.

These cases distinguish a circular cross-section from a rotated bounding box,
and separate object height, grasp width, and orientation-dependent floor support.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pressb.pick_place.task import (
    PickPlaceOutcome,
    cube_inside_box,
    object_geometry,
    object_inside_box,
)


IDENTITY = [1., 0., 0., 0.]


def quaternion(axis, angle):
    xyzw = Rotation.from_euler(axis, angle, degrees=True).as_quat()
    return xyzw[[3, 0, 1, 2]]


def box(xy=(.25, .25), *, yaw=0., floor=.005, rim=.145):
    return dict(center_world=[.4, -.2, 0.], inner_size_xy=list(xy),
                yaw_deg=yaw, floor_z_world=floor, rim_z_world=rim)


def cylinder(radius=.02, height=.06):
    return dict(shape="cylinder", radius=radius, height=height,
                size_xyz=[2*radius, 2*radius, height], grasp_width_m=2*radius)


def cuboid(size=(.08, .03, .02), grasp_width=.03):
    return dict(shape="cuboid", size_xyz=list(size), grasp_width_m=grasp_width)


def sample(checker, seconds, z, *, orientation=IDENTITY, width=.10, contact=False):
    return checker.update(seconds, [.4, -.2, z], orientation, np.zeros(6),
                          width, finger_contact=contact)


@pytest.mark.parametrize("angle", [0., 17., 45., 79.])
def test_spinning_upright_cylinder_does_not_expand_its_round_cross_section(angle):
    # 40 mm diameter fits inside 45 mm walls at every axial rotation. A rotated
    # 40x40 mm cuboid envelope would wrongly reject the 45-degree case.
    geometry = cylinder()
    container = box((.045, .045), rim=.075)
    assert object_inside_box([.4, -.2, .035], quaternion("z", angle), geometry, container)


def test_cylinder_radius_must_clear_the_wall_even_when_its_axis_is_inside():
    geometry = cylinder()
    container = box((.045, .045), rim=.075)
    assert object_inside_box([.402, -.2, .035], IDENTITY, geometry, container)
    assert not object_inside_box([.408, -.2, .035], IDENTITY, geometry, container)


def test_horizontal_cylinder_uses_its_length_along_the_rotated_box():
    geometry = cylinder(radius=.02, height=.08)
    # Align the 80 mm axis with a box yawed by 37 degrees, whose width is only
    # 45 mm. The same cylinder perpendicular to that box cannot fit.
    # Compose explicitly: the cylinder's local Z becomes world X, then yaws.
    rotation = Rotation.from_euler("z", 37., degrees=True) * Rotation.from_euler("y", 90., degrees=True)
    aligned = rotation.as_quat()[[3, 0, 1, 2]]
    wrong = (Rotation.from_euler("z", 127., degrees=True) * Rotation.from_euler("y", 90., degrees=True)).as_quat()[[3, 0, 1, 2]]
    container = box((.085, .045), yaw=37., rim=.05)
    assert object_inside_box([.4, -.2, .025], aligned, geometry, container)
    assert not object_inside_box([.4, -.2, .025], wrong, geometry, container)


def test_tilted_cylinder_must_clear_both_rim_and_floor():
    geometry = cylinder(radius=.02, height=.06)
    tilted = quaternion("x", 45.)
    # At this angle the 60 mm axial length and 40 mm diameter project to a
    # total height 50*sqrt(2) mm. The full cylinder, not just its centre, matters.
    support_height = .05 / np.sqrt(2)
    container = box(rim=.085)
    resting = container["floor_z_world"] + support_height
    assert object_inside_box([.4, -.2, resting], tilted, geometry, container)
    assert not object_inside_box([.4, -.2, resting-.006], tilted, geometry, container)
    assert not object_inside_box([.4, -.2, .06], tilted, geometry, container)


def test_non_cubic_object_uses_its_actual_dimensions_and_box_orientation():
    geometry = cuboid()
    container = box((.085, .035), yaw=90., rim=.05)
    assert object_inside_box([.4, -.2, .015], quaternion("z", 90.), geometry, container)
    assert not object_inside_box([.4, -.2, .015], IDENTITY, geometry, container)


@pytest.mark.parametrize("orientation,rest_z", [
    (quaternion("y", 90.), .025),
    (quaternion("x", 45.), .005 + .05/np.sqrt(2)),
])
def test_tipped_cylinder_can_finish_only_on_its_actual_lowest_surface(orientation, rest_z):
    geometry = cylinder()
    checker = PickPlaceOutcome(geometry, 0., box())
    sample(checker, 0., .25, width=.04)
    sample(checker, 1., rest_z+.012, orientation=orientation)
    suspended = sample(checker, 2., rest_z+.012, orientation=orientation)
    assert suspended["inside_box"] and suspended["object_stable"]
    assert not suspended["resting_on_box_floor"] and not suspended["success"]
    assert not sample(checker, 3., rest_z, orientation=orientation)["success"]
    resting = sample(checker, 3.5, rest_z, orientation=orientation)
    assert resting["resting_on_box_floor"] and resting["success"]


def test_tall_upright_cylinder_at_rest_does_not_satisfy_prior_lift():
    # Height is 120 mm but grasp diameter only 30 mm. Treating the diameter as
    # its vertical size would invent a 45 mm lift while it simply rests.
    geometry = cylinder(radius=.015, height=.12)
    checker = PickPlaceOutcome(geometry, 0., box())
    sample(checker, 0., .065)
    result = sample(checker, 1., .065)
    assert result["inside_box"] and result["resting_on_box_floor"]
    assert not result["lifted"] and not result["success"]


def test_tipping_a_long_flat_object_does_not_invent_lift_from_its_rising_centre():
    geometry = cuboid((.14, .03, .02))
    checker = PickPlaceOutcome(geometry, 0., box())
    # Both poses rest on the 5 mm-high box floor. Tipping the long side upward
    # raises the centre by 60 mm without lifting the lowest surface at all.
    sample(checker, 0., .015)
    upright = quaternion("y", 90.)
    sample(checker, 1., .075, orientation=upright)
    result = sample(checker, 2., .075, orientation=upright)
    assert result["inside_box"] and result["resting_on_box_floor"]
    assert result["max_lift_m"] > .04  # Preserved centre-based diagnostic.
    assert result["max_bottom_clearance_m"] == pytest.approx(.005)
    assert not result["lifted"] and not result["success"]


def test_release_uses_configured_grasp_width_and_requires_no_finger_contact():
    geometry = cuboid()
    checker = PickPlaceOutcome(geometry, 0., box())
    sample(checker, 0., .25)
    sample(checker, 1., .015, width=.04, contact=True)
    assert not sample(checker, 2., .015, width=.04, contact=True)["success"]
    assert not sample(checker, 3., .015, width=.04)["success"]
    # 40 mm clears the specified 30 mm grasp direction, although another
    # dimension of the object is 80 mm. Release is physical-direction specific.
    result = sample(checker, 3.5, .015, width=.04)
    assert result["gripper_released"] and result["success"]


@pytest.mark.parametrize("shape", ["cuboid", "cube"])
def test_legacy_cube_input_preserves_containment_and_success_contract(shape):
    edge = .035
    geometry = cuboid((edge, edge, edge), grasp_width=edge)
    # build_scene returns the legacy cube using this alias plus metadata that
    # is shared with its other primitive types.
    geometry.update(shape=shape, radius=None, height=edge)
    container = box((.09, .09), rim=.08)
    for position in ([.4, -.2, .0225], [.438, -.2, .04], [.4, -.2, .075]):
        for orientation in (IDENTITY, quaternion("z", 45.)):
            assert object_inside_box(position, orientation, geometry, container) == cube_inside_box(position, orientation, edge, container)
    old, generic = (PickPlaceOutcome(value, 0., container) for value in (edge, geometry))
    for seconds, z in ((0., .2), (1., .0225), (1.5, .0225)):
        legacy, current = sample(old, seconds, z), sample(generic, seconds, z)
        for field in ("success", "first_success_time_s", "inside_box", "resting_on_box_floor", "max_lift_m"):
            assert legacy[field] == current[field]
    assert current["success"]


@pytest.mark.parametrize("geometry", [
    {"shape": "cube", "edge": .03},
    {"shape": "cube", "size_xyz": [.03, .03, .03]},
    {"shape": "cube", "edge": .03, "size": [.03, .03, .03]},
])
def test_cube_geometry_uses_one_equal_edge_for_every_axis(geometry):
    normalized = object_geometry(geometry)
    assert normalized["size_xyz"] == [.03, .03, .03]
    assert normalized["edge"] == .03
    assert normalized["grasp_width_m"] == .03
    # A cube specification must retain the actual 30 mm wall clearance.
    assert object_inside_box([.4, -.2, .020], IDENTITY, normalized, box((.032, .032)))
    assert not object_inside_box([.404, -.2, .020], IDENTITY, normalized, box((.032, .032)))


@pytest.mark.parametrize("geometry", [
    {"shape": "cube", "size_xyz": [.03, .04, .03]},
    {"shape": "cube", "edge": .03, "size": [.03, .025, .03]},
    {"shape": "cube", "edge": .03, "size_xyz": [.04, .04, .04]},
    {"shape": "cube", "edge": .03, "size_xyz": [.03, .03, .03], "size": [.03, .04, .03]},
])
def test_cube_label_cannot_hide_a_rectangular_object_or_conflicting_edge(geometry):
    with pytest.raises(ValueError, match="Cube dimensions"):
        object_geometry(geometry)


def test_explicit_cube_scene_matches_legacy_cube_physics_and_geometry():
    """Run with PYTHONPATH=.cache/usd-inspect:src for isolated CPU USD."""
    Usd = pytest.importorskip("pxr.Usd")
    from pxr import UsdGeom, UsdPhysics
    from pressb.pick_place.scene import build_scene

    parameters = dict(edge=.03, center_xy=[.28, -.02], mass=.025, yaw_deg=17)
    common = dict(background=False, lights=False, mount_hardware=False)
    legacy_stage, explicit_stage = Usd.Stage.CreateInMemory(), Usd.Stage.CreateInMemory()
    legacy = build_scene(legacy_stage, {**common, "cube": parameters})
    explicit = build_scene(explicit_stage, {**common, "object": {"shape": "cube", **parameters}})
    assert legacy_stage.GetRootLayer().ExportToString() == explicit_stage.GetRootLayer().ExportToString()
    assert explicit["object_geometry"] == legacy["object_geometry"]
    assert explicit["cube_edge"] == .03
    assert explicit["object_geometry"]["edge"] == .03
    assert explicit["object_geometry"]["size_xyz"] == [.03, .03, .03]
    primitive = explicit_stage.GetPrimAtPath(explicit["object_path"])
    assert primitive.IsA(UsdGeom.Cube)
    assert primitive.HasAPI(UsdPhysics.CollisionAPI)
    assert primitive.HasAPI(UsdPhysics.RigidBodyAPI)
    np.testing.assert_allclose(UsdPhysics.MassAPI(primitive).GetDiagonalInertiaAttr().Get(),
                               [.025 * .03 ** 2 / 6] * 3, rtol=1e-6)


@pytest.mark.parametrize("object_config,match", [
    ({"shape": "cube", "size": [.03, .03, .03]}, "requires a single object.edge"),
    ({"shape": "cube", "edge": .03, "size": [.03, .025, .03]}, "must equal"),
    ({"shape": "cube", "edge": .03, "size_xyz": [.04, .04, .04]}, "must equal"),
])
def test_scene_rejects_missing_or_conflicting_cube_edge(object_config, match):
    Usd = pytest.importorskip("pxr.Usd")
    from pressb.pick_place.scene import build_scene

    stage = Usd.Stage.CreateInMemory()
    with pytest.raises(ValueError, match=match):
        build_scene(stage, {"object": object_config})
    assert not stage.GetPrimAtPath("/World/PickPlace")


@pytest.mark.parametrize("geometry,expected_size,expected_edge", [
    ({"shape": "cube", "edge": .03, "size": [.03] * 3}, [.03] * 3, .03),
    ({"shape": "cuboid", "size": [.04, .025, .03]}, [.04, .025, .03], None),
    ({"shape": "cylinder", "radius": .02, "height": .03}, [.04, .04, .03], None),
])
def test_scene_reports_shape_correct_dimensions_without_cube_alias_for_other_shapes(geometry, expected_size, expected_edge):
    Usd = pytest.importorskip("pxr.Usd")
    from pressb.pick_place.scene import build_scene

    result = build_scene(Usd.Stage.CreateInMemory(), {"object": geometry,
                         "background": False, "lights": False, "mount_hardware": False})
    assert result["object_geometry"]["shape"] == geometry["shape"]
    assert result["object_geometry"]["size_xyz"] == expected_size
    assert result["object_geometry"]["edge"] == expected_edge
    assert result["cube_edge"] == expected_edge
    assert object_geometry(result["object_geometry"])["size_xyz"] == expected_size
