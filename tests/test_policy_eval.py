"""Learned-action convention and projection checks without Isaac or a model."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pressb.kinematics import PiperKinematics
from pressb.policy_eval import InvalidPolicyAction, PolicyPoseController, validate_policy_reply

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/scene.json").read_text())
URDF = ROOT / CFG["robot_urdf"]


def pose(kin, q):
    matrix = kin.fk(q)
    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]
    return np.r_[matrix[:3, 3], quaternion, .008], np.r_[matrix[:3, 3], matrix[:2, :3].reshape(6)]


def test_reply_rejects_rotation_convention_and_gripper_changes():
    kin = PiperKinematics(URDF, tip_offset=.1358)
    eight, nine = pose(kin, [.2, .7, -1., .25, .3, -.2])
    reply = dict(actions_pose8=np.tile(eight, (7, 1)).tolist(), actions_pose9=np.tile(nine, (7, 1)).tolist())
    np.testing.assert_allclose(validate_policy_reply(reply)[0][0], eight)
    reply["actions_pose9"][0][3:] = kin.fk([.2, .7, -1., .25, .3, -.2])[:3, :2].reshape(6).tolist()
    with pytest.raises(InvalidPolicyAction, match="rotation6D"):
        validate_policy_reply(reply)
    reply["actions_pose9"][0] = nine.tolist()
    reply["actions_pose8"][0][7] = .02
    with pytest.raises(InvalidPolicyAction, match="gripper"):
        validate_policy_reply(reply)


def test_projected_repeated_target_keeps_moving_with_velocity_limits():
    kin = PiperKinematics(URDF, tip_offset=.1358)
    initial = np.array([.2, .7, -1., .25, .3, -.2])
    wanted = initial + [.4, -.2, .15, .3, -.3, .2]
    raw, _ = pose(kin, wanted)
    original = raw.copy()
    controller = PolicyPoseController(URDF, initial)
    first, diag1 = controller.solve(raw)
    second, diag2 = controller.solve(raw)
    np.testing.assert_array_equal(raw, original)
    assert np.linalg.norm(second - first) > .01
    for previous, actual in ((initial, first), (first, second)):
        assert np.all(np.abs(actual - previous) <= kin.velocity_limits / 30 + 1e-8)
        assert np.all(actual >= kin.lower) and np.all(actual <= kin.upper)
    assert diag1["velocity_saturated_joints"]
    assert diag1["command_position_residual_m"] > diag1["unrestricted_position_residual_m"]
    assert diag2["command_position_residual_m"] < diag1["command_position_residual_m"]


def test_exact_reachable_target_preserves_tcp_pose():
    kin = PiperKinematics(URDF, tip_offset=.1358)
    initial = np.array([.2, .7, -1., .25, .3, -.2])
    wanted = initial + [.01, -.01, .01, .01, -.01, .01]
    raw, _ = pose(kin, wanted)
    actual, diagnostics = PolicyPoseController(URDF, initial).solve(raw)
    np.testing.assert_allclose(kin.fk(actual), kin.fk(wanted), atol=1e-6)
    assert diagnostics["command_position_residual_m"] < 1e-6
