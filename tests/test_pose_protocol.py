"""Shared pose conversion gets gripper semantics from the caller."""
import numpy as np
import pytest

from pressb.control.pose_protocol import pose8_to_pose9, pose9_to_pose8


@pytest.mark.parametrize("width", [0., .042])
def test_pose_conversion_preserves_batched_transforms_with_the_requested_width(width):
    pose = np.array([[[.2, -.1, .4, 1., 0., 0., 0., .007],
                      [.3, .2, .5, .5, .5, .5, .5, .009]]])
    pose9 = pose8_to_pose9(pose)
    converted = pose9_to_pose8(pose9, gripper_width_m=width)
    assert converted.shape == pose.shape
    np.testing.assert_allclose(converted[..., :7], pose[..., :7], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(converted[..., 7], np.full((1, 2), width))
    np.testing.assert_allclose(pose8_to_pose9(converted), pose9, rtol=0, atol=1e-15)


@pytest.mark.parametrize("width", [-.001, np.nan, np.inf, -np.inf, True, ".008", [.008], None, 1j])
def test_invalid_gripper_width_is_rejected(width):
    pose = [.2, .1, .4, 1., 0., 0., 0., 1., 0.]
    with pytest.raises(ValueError, match="gripper_width_m"):
        pose9_to_pose8(pose, gripper_width_m=width)


def test_gripper_width_has_no_implicit_default_and_is_keyword_only():
    pose = [.2, .1, .4, 1., 0., 0., 0., 1., 0.]
    with pytest.raises(TypeError, match="gripper_width_m"):
        pose9_to_pose8(pose)
    with pytest.raises(TypeError):
        pose9_to_pose8(pose, .008)
