"""Full-pose, bounded-branch and causal-time tests for action-only replay IK."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pressb.kinematics import PiperKinematics
from pressb.replay_control import ContinuousPoseIK, ReplayIKError, action_pose, solve_action_trajectory


ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/scene.json").read_text())
URDF = ROOT / CFG["robot_urdf"]


def action_for(kin, q, width=.008):
    pose = kin.fk(q)
    xyzw = Rotation.from_matrix(pose[:3, :3]).as_quat()
    return np.r_[pose[:3, 3], xyzw[[3, 0, 1, 2]], width]


@pytest.fixture
def kin():
    return PiperKinematics(URDF, tip_offset=.1358)


def test_replay_controls_full_pose_at_correct_tcp_offset(kin):
    seed = np.array([.2, .7, -1., .25, .3, -.2])
    target = seed + [.012, -.01, .015, .02, -.012, .016]
    action = action_for(kin, target)
    solver = ContinuousPoseIK(URDF, seed)
    result = solver.solve(action)
    actual, wanted = kin.fk(result), kin.fk(target)
    np.testing.assert_allclose(actual[:3, 3], wanted[:3, 3], atol=1e-6)
    assert Rotation.from_matrix(wanted[:3, :3] @ actual[:3, :3].T).magnitude() < 1e-5
    assert np.linalg.norm(PiperKinematics(URDF, tip_offset=.24).fk(result)[:3, 3] - action[:3]) > .1


def test_quaternion_sign_equivalence_and_float32_rounding(kin):
    seed = np.array([.2, .7, -1., .25, .3, -.2])
    solver = ContinuousPoseIK(URDF, seed)
    action = action_for(kin, seed).astype(np.float32)
    first = solver.solve(action)
    action[3:7] *= -1
    np.testing.assert_array_equal(solver.solve(action), first)
    for bad in (np.zeros(4), action[3:7] * 2, np.full(4, np.nan)):
        malformed = action.copy()
        malformed[3:7] = bad
        with pytest.raises(ValueError):
            action_pose(malformed)


def test_continuous_path_stays_in_previous_branch_and_velocity_bounds(kin):
    seed = np.array([.2, .7, -1., .25, .3, -.2])
    solver = ContinuousPoseIK(URDF, seed)
    last = seed
    for phase in np.linspace(0., 1., 30):
        known = seed + phase * np.array([.1, -.06, .09, .11, -.09, .08])
        solved = solver.solve(action_for(kin, known))
        assert np.all(solved >= kin.lower) and np.all(solved <= kin.upper)
        assert np.all(np.abs(solved - last) <= kin.velocity_limits / 30 + 1e-9)
        np.testing.assert_allclose(solved, known, atol=5e-5)
        last = solved


def test_unreachable_or_too_fast_target_fails_without_changing_seed(kin):
    seed = np.array([.2, .7, -1., .25, .3, -.2])
    solver = ContinuousPoseIK(URDF, seed)
    action = action_for(kin, seed)
    action[0] += 10.
    with pytest.raises(ReplayIKError):
        solver.solve(action)
    np.testing.assert_array_equal(solver.q, seed)
    # This pose is reachable globally, but not continuously within one sample.
    too_fast = seed.copy()
    too_fast[0] += .5
    with pytest.raises(ReplayIKError):
        solver.solve(action_for(kin, too_fast))
    np.testing.assert_array_equal(solver.q, seed)
    with pytest.raises(ValueError, match="joint limits"):
        ContinuousPoseIK(URDF, kin.upper + .1)
    action = action_for(kin, seed, width=.1)
    with pytest.raises(ValueError, match="finger limits"):
        solver.solve(action)


def test_next_sample_targets_are_reached_after_four_steps_and_terminal_is_not_extended(kin):
    seed = np.array([.2, .7, -1., .25, .3, -.2])
    q1 = seed + [.01, -.01, .01, .01, -.01, .01]
    q2 = seed + [.02, -.02, .02, .02, -.02, .02]
    actions = np.vstack([action_for(kin, q1), action_for(kin, q2), action_for(kin, q2)])
    plan = solve_action_trajectory(URDF, actions, seed)
    assert plan.joint_positions.shape == (9, 6)
    np.testing.assert_array_equal(plan.joint_positions[0], seed)
    np.testing.assert_allclose(plan.joint_positions[4], q1, atol=5e-5)
    np.testing.assert_allclose(plan.joint_positions[8], q2, atol=5e-5)
    np.testing.assert_allclose(plan.joint_positions[1], .75 * seed + .25 * plan.ik_solutions[0])
    np.testing.assert_array_equal(plan.action_index_by_step, [-1, 0, 0, 0, 0, 1, 1, 1, 1])
    assert plan.times_s[-1] == 2 / 30
    actions[-1] = action_for(kin, q1)
    with pytest.raises(ValueError, match="terminal-clamped"):
        solve_action_trajectory(URDF, actions, seed)


def test_total_gripper_width_is_interpolated_and_velocity_bounded(kin):
    seed = np.array([.2, .7, -1., .25, .3, -.2])
    action = action_for(kin, seed, width=.012)
    plan = solve_action_trajectory(URDF, [action, action], seed, initial_gripper_width=.008)
    np.testing.assert_allclose(plan.gripper_widths, [.008, .009, .010, .011, .012])
    action[7] = .07
    with pytest.raises(ReplayIKError, match="gripper width"):
        solve_action_trajectory(URDF, [action, action], seed, fps=120, initial_gripper_width=.008)
