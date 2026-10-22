"""Causal arm smoothing must preserve gripper timing and policy chunk continuity."""
import numpy as np
import pytest

from pressb.action_postprocessing import JointActionPostprocessor


def test_first_action_starts_from_measured_state_and_gripper_is_unfiltered():
    processor = JointActionPostprocessor("joint_ema", .8)
    initial = np.r_[np.full(6, .5), .07]
    command = np.r_[np.full(6, 1.5), 0.]
    processor.reset(initial)
    initial[:] = -3.  # The caller's input cannot change the saved initial pose.
    sent = processor.apply(command)
    np.testing.assert_allclose(sent[:6], 1.3)
    assert sent[6] == command[6] == 0.
    np.testing.assert_array_equal(command, np.r_[np.full(6, 1.5), 0.])


def test_chunk_boundary_keeps_previous_sent_action_and_outputs_do_not_alias_state():
    processor = JointActionPostprocessor("joint_ema", .5)
    processor.reset(np.zeros(7))
    first_chunk = [np.r_[np.full(6, 1.), .08], np.r_[np.full(6, 1.), .03]]
    sent = [processor.apply(command) for command in first_chunk]
    np.testing.assert_allclose(sent[-1][:6], .75)
    sent[-1][:] = 999.  # Does not corrupt the retained filter state.
    next_chunk_first = processor.apply(np.r_[np.full(6, -1.), .09])
    np.testing.assert_allclose(next_chunk_first[:6], -.125)
    assert next_chunk_first[6] == .09
    assert processor.summary()["applied_actions"] == 3
    assert processor.summary()["control_hz"] == 30
    assert processor.summary()["gripper_filtered"] is False


@pytest.mark.parametrize("mode,alpha", [("none", .8), ("joint_ema", 1.)])
def test_none_and_alpha_one_are_exact_pass_through(mode, alpha):
    processor = JointActionPostprocessor(mode, alpha)
    if mode == "joint_ema":
        processor.reset(np.full(7, -2.))
    raw = np.array([-.0, .1234567, -.9876543, 1., 2., -3., .03123456], np.float32)
    sent = processor.apply(raw)
    assert sent.dtype == raw.dtype and sent.tobytes() == raw.tobytes()
    assert not np.shares_memory(sent, raw)


@pytest.mark.parametrize("alpha", [0., -1., 1.01, np.nan, np.inf, True, "0.8"])
def test_invalid_alpha_is_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        JointActionPostprocessor(alpha=alpha)


@pytest.mark.parametrize("alpha", [.8, 1.])
def test_ema_requires_reset_even_when_alpha_is_one(alpha):
    with pytest.raises(RuntimeError, match="Reset"):
        JointActionPostprocessor("joint_ema", alpha).apply(np.zeros(7))


@pytest.mark.parametrize("invalid", [np.zeros(6), np.zeros((1, 7)), np.full(7, np.nan), np.full(7, 1j)])
def test_invalid_input_is_rejected_without_mutating_state(invalid):
    processor = JointActionPostprocessor("joint_ema", .5)
    processor.reset(np.zeros(7))
    with pytest.raises(ValueError, match="seven finite real"):
        processor.apply(invalid)
    with pytest.raises(ValueError, match="seven finite real"):
        processor.reset(invalid)
    np.testing.assert_array_equal(processor.apply(np.ones(7)), np.r_[np.full(6, .5), 1.])


def test_mode_is_validated_and_reset_starts_a_new_episode():
    with pytest.raises(ValueError, match="mode"):
        JointActionPostprocessor("xyz_ema")
    processor = JointActionPostprocessor("joint_ema", .5)
    processor.reset(np.zeros(7))
    processor.apply(np.ones(7))
    processor.reset(np.full(7, 2.))
    np.testing.assert_array_equal(processor.apply(np.zeros(7)), np.r_[np.ones(6), 0.])
    assert processor.summary()["applied_actions"] == 1
