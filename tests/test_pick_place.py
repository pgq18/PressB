"""CPU contracts for physical PAP placement and recorded command semantics.

The geometric cases deliberately put a cube's centre inside the container while
its rotated corners cross a wall, floor, or rim. A centre-only predicate would
accept these invalid placements.
"""
from copy import deepcopy

import numpy as np
import pytest

from pressb.pick_place.task import PickPlaceOutcome, cube_inside_box, load_trajectory


EDGE = .035
BOX = dict(center_world=[.32, .21, .76], inner_size_xy=[.17, .15],
           yaw_deg=0., floor_z_world=.765, rim_z_world=.835)
REST = [.32, .21, .7825]
IDENTITY = [1., 0., 0., 0.]


def yaw(angle):
    half = np.deg2rad(angle) / 2
    return [float(np.cos(half)), 0., 0., float(np.sin(half))]


def update(checker, seconds, *, position=REST, quaternion=IDENTITY,
           velocity=(0., 0., 0., 0., 0., 0.), width=.10, finger_contact=False):
    return checker.update(seconds, position, quaternion, velocity, width,
                          finger_contact=finger_contact)


def lifted_checker():
    checker = PickPlaceOutcome(EDGE, .76, deepcopy(BOX), stable_seconds=.5)
    # An actual high object pose occurs before the placement, outside the box.
    update(checker, 0., position=[.314, .006, .86], width=.035)
    return checker


def test_rotation_can_make_a_centre_inside_placement_cross_a_wall():
    position = [.385, .21, .79]
    assert cube_inside_box(position, IDENTITY, EDGE, BOX)
    assert not cube_inside_box(position, yaw(45), EDGE, BOX)


def test_rotated_box_uses_its_local_walls_not_its_world_bounding_rectangle():
    box = dict(center_world=[1., -2., .76], inner_size_xy=[.14, .06],
               yaw_deg=90., floor_z_world=.765, rim_z_world=.835)
    assert cube_inside_box([1., -1.955, .79], yaw(90), EDGE, box)
    assert not cube_inside_box([1.02, -2., .79], yaw(90), EDGE, box)


@pytest.mark.parametrize("position,quaternion", [
    ([.32, .21, .77], IDENTITY),  # Centre above floor, bottom below floor.
    ([.32, .21, .83], IDENTITY),  # Centre below rim, top above rim.
    ([.32, .21, .785], [np.cos(np.pi/8), np.sin(np.pi/8), 0., 0.]),
])
def test_every_corner_must_clear_floor_and_rim(position, quaternion):
    assert not cube_inside_box(position, quaternion, EDGE, BOX)


@pytest.mark.parametrize("quaternion", [[0., 0., 0., 0.], [2., 0., 0., 0.], [np.nan, 0., 0., 0.]])
def test_invalid_orientation_never_counts_as_containment(quaternion):
    assert not cube_inside_box(REST, quaternion, EDGE, BOX)


def test_resting_in_box_without_prior_lift_is_not_a_pick_and_place_success():
    checker = PickPlaceOutcome(EDGE, .76, BOX)
    update(checker, 0.)
    result = update(checker, 2.)
    assert result["inside_box"] and result["object_stable"]
    assert not result["lifted"] and not result["success"]


def test_closed_gripper_does_not_start_the_release_dwell():
    checker = lifted_checker()
    update(checker, 1., width=.035)
    assert not update(checker, 4., width=.035)["success"]
    assert not update(checker, 5.)["success"]
    assert not update(checker, 5.49)["success"]
    assert update(checker, 5.5)["success"]


@pytest.mark.parametrize("interruption", [
    dict(position=[.32, .31, .7825]),
    dict(velocity=(.1, 0., 0., 0., 0., 0.)),
    dict(velocity=(0., 0., 0., 0., 0., 1.)),
    dict(width=.035),
])
def test_containment_release_and_stability_must_be_continuous(interruption):
    checker = lifted_checker()
    update(checker, 1.)
    update(checker, 1.4, **interruption)
    assert not update(checker, 1.5)["success"]
    assert not update(checker, 1.99)["success"]
    assert update(checker, 2.)["success"]


def test_success_records_first_dwell_completion_time():
    checker = lifted_checker()
    update(checker, 1.)
    first = update(checker, 1.5)
    assert first["success"]
    assert first["first_success_time_s"] == 1.5
    assert update(checker, 2.)["first_success_time_s"] == 1.5


def test_motionless_cube_suspended_inside_cavity_is_not_placed():
    checker = lifted_checker()
    # Entire cube fits below the rim, but its bottom is 27.5 mm above the box
    # floor. This can occur when a cube is still supported by a gripper.
    update(checker, 1., position=[.32, .21, .81])
    result = update(checker, 2., position=[.32, .21, .81])
    assert result["inside_box"] and result["object_stable"]
    assert not result["success"]


def test_gripper_contact_prevents_success_even_when_width_looks_released():
    checker = lifted_checker()
    # A 45-degree cube spans almost 49.5 mm between fingers: a scalar width
    # threshold based only on its 35 mm edge cannot prove physical release.
    update(checker, 1., quaternion=yaw(45), width=.049, finger_contact=True)
    result = update(checker, 2., quaternion=yaw(45), width=.049, finger_contact=True)
    assert result["inside_box"] and result["object_stable"]
    assert not result["gripper_released"] and not result["success"]
    assert not update(checker, 2.1, quaternion=yaw(45))["success"]
    assert update(checker, 2.6, quaternion=yaw(45))["success"]


def test_rolled_cube_can_rest_on_its_lowest_edge_instead_of_an_axis_aligned_face():
    checker = lifted_checker()
    quaternion = [np.cos(np.pi/8), np.sin(np.pi/8), 0., 0.]
    # A 45-degree roll raises the centre by edge/sqrt(2) above the support.
    position = [.32, .21, BOX["floor_z_world"] + EDGE/np.sqrt(2)]
    update(checker, 1., position=position, quaternion=quaternion)
    result = update(checker, 1.5, position=position, quaternion=quaternion)
    assert result["resting_on_box_floor"] and result["success"]


def save_trajectory(tmp_path, *, widths=(10., 4.), state_width=8., times=(14., 14.5)):
    state = np.array([[0., 90., -180., 45., -30., 360., state_width],
                      [10., 70., -100., 40., -20., 180., 3.]])
    action = np.array([[180., -90., 0., 30., -45., 720., widths[0]],
                       [-180., 90., 90., 60., -90., 450., widths[1]]])
    path = tmp_path / "recorded.npz"
    np.savez(path, state_raw=state, action_raw=action, timestamp_s=np.asarray(times))
    return path, state, action


def test_loader_converts_recorded_units_without_clipping_or_wrapping_arm_angles(tmp_path):
    path, _, _ = save_trajectory(tmp_path)
    replay = load_trajectory(path, source="action", maximum_width_m=.1)
    np.testing.assert_allclose(replay["commands"][0], [np.pi, -np.pi/2, 0., np.pi/6, -np.pi/4, 4*np.pi, .1])
    np.testing.assert_allclose(replay["times"], [0., .5])
    assert replay["metadata"]["width_saturated_samples"] == 0
    assert replay["metadata"]["source_angle_unit"] == "degree"
    assert replay["metadata"]["source_width_unit"] == "cm"


def test_action_replay_still_initializes_from_the_measured_observation(tmp_path):
    path, state, action = save_trajectory(tmp_path)
    replay = load_trajectory(path, source="action", maximum_width_m=.1)
    np.testing.assert_allclose(replay["initial"][:6], np.deg2rad(state[0, :6]))
    assert replay["initial"][-1] == pytest.approx(.08)
    assert not np.allclose(replay["initial"][:6], np.deg2rad(action[0, :6]))
    observed = load_trajectory(path, source="observation", maximum_width_m=.1)
    np.testing.assert_allclose(observed["commands"][0], replay["initial"])


def test_saturating_invalid_recorded_gripper_width_is_explicit(tmp_path):
    path, _, _ = save_trajectory(tmp_path, widths=(13., -2.))
    replay = load_trajectory(path, source="action", maximum_width_m=.1)
    np.testing.assert_allclose(replay["commands"][:, -1], [.1, 0.])
    metadata = replay["metadata"]
    assert metadata["width_saturated_samples"] == 2
    assert metadata["max_width_saturation_m"] == pytest.approx(.03)
    assert metadata["raw_width_range_m"] == [-.02, .13]


def test_default_replay_selects_observed_arm_and_commanded_gripper_independently(tmp_path):
    path, state, action = save_trajectory(tmp_path)
    original_bytes = path.read_bytes()
    replay = load_trajectory(path)
    np.testing.assert_allclose(replay["commands"][:, :6], np.deg2rad(state[:, :6]))
    np.testing.assert_allclose(replay["commands"][:, 6], action[:, 6] / 100)
    assert replay["initial"][-1] == pytest.approx(.08)
    assert replay["metadata"]["source"] == "observed_arm_action_gripper"
    assert replay["metadata"]["arm_source"] == "observation.state"
    assert replay["metadata"]["gripper_source"] == "action"
    assert path.read_bytes() == original_bytes

    # Changing data that this mode does not select must leave its commands
    # unchanged; the observed first gripper width still defines the reset.
    state[:, 6] = [.01, .02]
    action[:, :6] += 27.
    altered = tmp_path / "altered.npz"
    np.savez(altered, state_raw=state, action_raw=action, timestamp_s=[14., 14.5])
    different = load_trajectory(altered)
    np.testing.assert_array_equal(different["commands"], replay["commands"])
    assert different["initial"][-1] == pytest.approx(.0001)


def test_default_replay_does_not_move_arm_to_an_unobserved_zero_action_prefix(tmp_path):
    state = np.tile([6., 38., -25., -9., 40., 10., .02], (20, 1))
    action = np.zeros_like(state)
    action[18:, :] = [5.8, 38., -24.7, -9.5, 38.9, 9.8, 5.6]
    path = tmp_path / "startup.npz"
    np.savez(path, state_raw=state, action_raw=action, timestamp_s=np.arange(20)/30)
    replay = load_trajectory(path)
    np.testing.assert_allclose(replay["commands"][:, :6], np.deg2rad(state[:, :6]))
    np.testing.assert_array_equal(replay["commands"][:18, 6], np.zeros(18))
    np.testing.assert_allclose(replay["commands"][18:, 6], [.056, .056])


@pytest.mark.parametrize("times", [(1., 1.), (2., 1.), (0., np.nan)])
def test_loader_rejects_timestamps_that_cannot_define_physical_replay(tmp_path, times):
    path, _, _ = save_trajectory(tmp_path, times=times)
    with pytest.raises(ValueError, match="Timestamps"):
        load_trajectory(path)
