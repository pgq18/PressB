"""Exact old/new outcome comparisons at geometry, timing and release boundaries."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pressb.tasks.pap.task import PickPlaceOutcome, _half_extent, object_geometry


SPEC = importlib.util.spec_from_file_location("pap_outcome_benchmark",
    Path(__file__).resolve().parents[1] / "scripts/benchmark_pick_place_outcome.py")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)
Reference = benchmark.ReferencePickPlaceOutcome
SHAPES = [.03, {"shape": "cube", "edge": .03},
          {"shape": "cuboid", "size_xyz": [.065, .027, .021], "grasp_width_m": .027},
          {"shape": "cylinder", "radius": .017, "height": .065}]


def container():
    return dict(center_world=[.4, -.2, .76], inner_size_xy=[.15, .13],
                floor_z_world=.765, rim_z_world=.87, yaw_deg=37.)


def compare(checkers, *args, **kwargs):
    expected, actual = (checker.update(*args, **kwargs) for checker in checkers)
    # Every numeric diagnostic and boolean must match exactly, not just success
    # or an allclose tolerance that could hide a changed threshold decision.
    assert actual == expected
    assert checkers[0].stable_since == checkers[1].stable_since
    return actual


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_random_rotations_and_all_cavity_boundaries_match_exactly(shape, dtype):
    rng = np.random.default_rng(901)
    geometry = object_geometry(shape)
    box = container()
    checkers = [cls(shape, .76, box) for cls in (Reference, PickPlaceOutcome)]
    for index in range(1500):
        if index % 25 == 0:
            # Exercise cache invalidation and translated/rotated boxes, too.
            box["yaw_deg"] = float(rng.uniform(-180, 180))
            box["center_world"][:2] = rng.uniform(-30, 30, 2).tolist()
        quat = Rotation.random(random_state=rng).as_quat()[[3, 0, 1, 2]].astype(dtype)
        rotation = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        inverse = Rotation.from_euler("z", -box["yaw_deg"], degrees=True).as_matrix()
        extent = _half_extent(rotation, geometry, inverse)
        bottom_extent = float(_half_extent(rotation, geometry, [0., 0., 1.]))
        position = np.asarray(box["center_world"], dtype=float)
        position[2] = box["floor_z_world"]+bottom_extent
        edge = index % 6
        epsilon = (-1, 0, 1)[(index//6) % 3]*1e-14
        if edge < 4:
            axis, sign = edge//2, (-1, 1)[edge % 2]
            local = np.zeros(3)
            local[axis] = sign*(box["inner_size_xy"][axis]/2+.001-extent[axis]+epsilon)
            position += inverse.T @ local
        elif edge == 4:
            position[2] = box["floor_z_world"]-.002+extent[2]+epsilon
        else:
            position[2] = box["rim_z_world"]+.001-extent[2]+epsilon
        velocity = rng.normal(0, .01, 6).astype(dtype)
        width = geometry["grasp_width_m"]+.008+epsilon
        compare(checkers, index/120, position.astype(dtype), quat, velocity, width,
                finger_contact=bool(index % 7 == 0))


@pytest.mark.parametrize("shape", SHAPES)
def test_full_lift_dwell_interruption_and_latched_success_sequences_match(shape):
    box = container()
    geometry = object_geometry(shape)
    rotation = Rotation.from_euler("xyz", [19, 41, 17], degrees=True)
    quat = rotation.as_quat()[[3, 0, 1, 2]]
    height = float(_half_extent(rotation.as_matrix(), geometry, [0., 0., 1.]))
    released = geometry["grasp_width_m"]+.008
    for condition in ("linear_speed", "angular_speed", "width", "contact", "floor", "lift"):
        checkers = [cls(shape, .76, deepcopy(box)) for cls in (Reference, PickPlaceOutcome)]
        kwargs = dict(position=[.4, -.2, box["floor_z_world"]+height], quaternion=quat,
                      velocity=np.zeros(6), gripper_width=released)
        compare(checkers, 0., [0., 0., .76+height+.04], quat, np.zeros(6), released)
        assert not compare(checkers, 1., **kwargs)["success"]
        interrupted = deepcopy(kwargs)
        if condition == "linear_speed":
            interrupted["velocity"][0] = .025
        elif condition == "angular_speed":
            interrupted["velocity"][3] = .25
        elif condition == "width":
            interrupted["gripper_width"] = np.nextafter(released, -np.inf)
        elif condition == "contact":
            interrupted["finger_contact"] = True
        elif condition == "floor":
            interrupted["position"][2] += .0030000001
        else:
            interrupted["position"][2] += .3
        assert not compare(checkers, 1.25, **interrupted)["success"]
        assert not compare(checkers, 2., **kwargs)["success"]
        assert not compare(checkers, np.nextafter(2.5, -np.inf), **kwargs)["success"]
        success = compare(checkers, 2.5, **kwargs)
        assert success["success"] and success["first_success_time_s"] == 2.5
        latched = compare(checkers, 3., **interrupted)
        assert latched["success"] and latched["first_success_time_s"] == 2.5


@pytest.mark.parametrize("shape", SHAPES)
def test_rest_lift_speed_release_and_quaternion_threshold_neighbors(shape):
    geometry, box = object_geometry(shape), container()
    for threshold_name, threshold in (("floor", .003), ("lift", .04), ("linear", .025),
                                      ("angular", .25), ("release", geometry["grasp_width_m"]+.008),
                                      ("quaternion_norm", 1.01)):
        for value in (np.nextafter(threshold, -np.inf), threshold, np.nextafter(threshold, np.inf)):
            checkers = [cls(shape, .76, box) for cls in (Reference, PickPlaceOutcome)]
            quat, velocity = np.array([1., 0., 0., 0.]), np.zeros(6)
            position = [.4, -.2, box["floor_z_world"]+geometry["size_xyz"][2]/2]
            width = geometry["grasp_width_m"]+.008
            lift = value if threshold_name == "lift" else .08
            compare(checkers, 0., [.4, -.2, .76+geometry["size_xyz"][2]/2+lift], quat, velocity, width)
            if threshold_name == "floor":
                position[2] += value
            elif threshold_name == "linear":
                velocity[0] = value
            elif threshold_name == "angular":
                velocity[3] = value
            elif threshold_name == "release":
                width = value
            elif threshold_name == "quaternion_norm":
                quat[0] = value
            for seconds in (1., 1.499999999, 1.5, 2.):
                compare(checkers, seconds, position, quat, velocity, width)


def test_box_mapping_changes_remain_visible_after_cache_creation():
    box = container()
    checkers = [cls(.03, .76, box) for cls in (Reference, PickPlaceOutcome)]
    args = ([.4, -.2, .78], [1., 0., 0., 0.], np.zeros(6), .1)
    compare(checkers, 0., *args)
    for index, (key, value) in enumerate((("center_world", [.35, -.2, .76]), ("yaw_deg", 76.),
                                        ("inner_size_xy", [.065, .05]), ("floor_z_world", .77),
                                        ("rim_z_world", .79))):
        box[key] = value
        compare(checkers, index+1, *args)
