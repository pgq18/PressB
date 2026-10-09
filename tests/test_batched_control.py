"""Geometry and actuator bounds for the fast vector controller."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pressb.kinematics import PiperKinematics
from pressb.online_rl.batched_control import BatchedPoseController

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / json.loads((ROOT / "configs/scene.json").read_text())["robot_urdf"]


def actions(transforms):
    quaternion = Rotation.from_matrix(transforms[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
    return np.c_[transforms[:, :3, 3], quaternion, np.full(len(transforms), .008)]


def test_batch_fk_and_analytic_jacobian_match_independent_scalar_and_finite_difference():
    rng = np.random.default_rng(8)
    kin = PiperKinematics(URDF, tip_offset=.1358)
    q = rng.uniform(kin.lower + .05, kin.upper - .05, (17, 6))
    controller = BatchedPoseController(URDF, q)
    transform, jac = controller.fk(q, jacobian=True)
    np.testing.assert_allclose(transform, np.stack([kin.fk(row) for row in q]), atol=1e-12)
    for joint in range(6):
        next_q = q.copy()
        next_q[:, joint] += 1e-6
        actual = controller.fk(next_q)
        np.testing.assert_allclose((actual[:, :3, 3] - transform[:, :3, 3]) / 1e-6,
                                   jac[:, :3, joint], atol=5e-7)
        angular = Rotation.from_matrix(actual[:, :3, :3] @ transform[:, :3, :3].transpose(0, 2, 1)).as_rotvec() / 1e-6
        np.testing.assert_allclose(angular, jac[:, 3:, joint], atol=1e-7)


def test_reachable_targets_and_subset_state_are_correct():
    q = np.tile([.2, .7, -1., .25, .3, -.2], (9, 1))
    controller = BatchedPoseController(URDF, q, iterations=20)
    ids = np.array([1, 3, 7])
    wanted = q[ids] + [.01, -.01, .01, .01, -.01, .01]
    target = controller.fk(wanted)
    result, diagnostics = controller.solve(ids, actions(target))
    np.testing.assert_allclose(controller.fk(result), target, atol=1e-6)
    np.testing.assert_array_equal(controller.q[[0, 2, 4, 5, 6, 8]], q[[0, 2, 4, 5, 6, 8]])
    assert max(d['command_position_residual_m'] for d in diagnostics) < 1e-6


def test_unreachable_targets_preserve_velocity_joint_bounds_and_improve_repeated_projection():
    q = np.tile([.2, .7, -1., .25, .3, -.2], (64, 1))
    controller = BatchedPoseController(URDF, q)
    target = controller.fk(q + [.4, -.2, .15, .3, -.3, .2])
    act = actions(target)
    ids = np.arange(64)
    previous = q
    costs = []
    for _ in range(5):
        result, diagnostics = controller.solve(ids, act)
        assert np.all(np.abs(result - previous) <= controller.maximum_joint_step + 1e-10)
        assert np.all(result >= controller.kinematics.lower)
        assert np.all(result <= controller.kinematics.upper)
        costs.append(diagnostics[0]['command_position_residual_m'] ** 2 +
                     (.15 * diagnostics[0]['command_rotation_residual_rad']) ** 2)
        previous = result
    assert costs[-1] < costs[0] * .1
    assert all(b <= a + 1e-12 for a, b in zip(costs, costs[1:]))


def test_reset_rejects_invalid_input_without_changing_other_environments():
    q = np.tile([.2, .7, -1., .25, .3, -.2], (3, 1))
    controller = BatchedPoseController(URDF, q)
    controller.reset([1], q[[1]] + .01)
    np.testing.assert_array_equal(controller.q[[0, 2]], q[[0, 2]])
    with pytest.raises(ValueError):
        controller.reset([0, 0], q[:2])
    invalid = actions(controller.fk(q))
    invalid[1, 7] = .02
    before = controller.q.copy()
    with pytest.raises(ValueError):
        controller.solve([0, 1, 2], invalid)
    np.testing.assert_array_equal(controller.q, before)
