"""Reference queue plus real PiPER adapter, using only a deterministic fake SDK.

These tests inspect the actual CAN-facing method arguments. They do not model
contact, physical motor dynamics, or successful grasping.
"""
import copy
import json
import threading

import numpy as np
import pytest

from pressb.control.real_replay import ReplayError, ReplayPlan, execute_replay
from pressb.robots.piper.real import PiperFeedbackError, PiperHardware, PiperHardwareError
from test_piper_real import Clock, SDK


MEASURED_RAW = np.array([12518, 136298, -107047, 301, 76276, 7556, 35100])
SDK_SCALE = np.r_[np.full(6, 180000 / np.pi), 1_000_000.]
LOWER = np.r_[np.deg2rad([-150, 0, -170, -100, -70, -170]), 0.]
UPPER = np.r_[np.deg2rad([150, 180, 0, 100, 70, 170]), .07]


def make_reference(**overrides):
    clock = Clock()
    sdk = SDK(clock)
    sdk.values = MEASURED_RAW.copy()
    config = dict(actuation_enabled=True, command_policy="reference", speed_percent=5,
                  joint_lower_rad=LOWER[:6], joint_upper_rad=UPPER[:6])
    config.update(overrides)
    robot = PiperHardware(config, sdk_factory=lambda _: sdk, can_probe=lambda _: None,
                          monotonic=clock.monotonic, wall_time=clock.wall, sleep=clock.sleep)
    return robot, sdk, clock


def joint_calls(sdk):
    return [row[1:] for row in sdk.calls if row[0] == "joints"]


def assert_targets_legal(sdk):
    for raw in joint_calls(sdk):
        assert np.all(np.asarray(raw) >= np.ceil(LOWER[:6] * SDK_SCALE[:6]))
        assert np.all(np.asarray(raw) <= np.floor(UPPER[:6] * SDK_SCALE[:6]))


def legal_target():
    value = MEASURED_RAW / SDK_SCALE
    value[4] = np.deg2rad(60)
    return value


def replay_plan():
    first = legal_target()
    first[0] = np.deg2rad(120)
    first[6] = .07
    second = first.copy()
    second[0] = np.deg2rad(-120)
    second[4] = 0.
    second[6] = 0.
    return dict(schema_version=1, source={"kind": "fake SDK adapter integration"}, fps=30,
                expected_initial_state=(MEASURED_RAW / SDK_SCALE).tolist(),
                expected_initial_tolerance=[np.deg2rad(.25)] * 6 + [.0005],
                actions=[first.tolist(), second.tolist()],
                robot=dict(actuation_enabled=True, speed_percent=5,
                           joint_lower_rad=LOWER[:6].tolist(), joint_upper_rad=UPPER[:6].tolist()),
                replay=dict(execution_mode="reference", max_wall_time_s=3))


def execute_actual_adapter(plan, path, configure=lambda sdk: None, **kwargs):
    clock = Clock()
    sdk = SDK(clock)
    sdk.values = MEASURED_RAW.copy()
    configure(sdk)
    constructed = []
    def factory(config):
        robot = PiperHardware(config, sdk_factory=lambda _: sdk, can_probe=lambda _: None,
                              monotonic=clock.monotonic, wall_time=clock.wall, sleep=clock.sleep)
        constructed.append(robot)
        return robot
    result = execute_replay(plan, factory, path, monotonic=clock.monotonic,
                            wall_time=clock.wall, sleep=clock.sleep, **kwargs)
    return result, constructed[0], sdk, clock


@pytest.mark.parametrize("speed_percent", [1, 2])
def test_outside_j5_enables_before_first_legal_inward_command_without_slew(speed_percent):
    robot, sdk, _ = make_reference(speed_percent=speed_percent)
    robot.connect(expected_state=MEASURED_RAW / SDK_SCALE, expected_tolerance=np.full(7, .001))
    assert sdk.motor_enabled == [True] * 6
    assert not joint_calls(sdk)
    applied = robot.command(legal_target())
    np.testing.assert_allclose(applied, np.rint(legal_target() * SDK_SCALE) / SDK_SCALE)
    assert joint_calls(sdk)[0][4] == 60000
    assert sdk.calls.index(("enable", 7)) < next(i for i, c in enumerate(sdk.calls) if c[0] == "joints")
    assert all(c == ("mode", 1, 1, speed_percent, 0) for c in sdk.calls if c[0] == "mode")
    assert_targets_legal(sdk)
    robot.close()


def test_actual_adapter_keeps_original_fresh_observation_delta_and_fixed_speed(tmp_path):
    result, robot, sdk, _ = execute_actual_adapter(replay_plan(), tmp_path)
    commands = np.asarray(joint_calls(sdk))
    # Measured state deliberately never follows the command: both targets must
    # use the unchanged feedback, not the preceding command or a slew baseline.
    assert commands[0, 0] == 12518 + 45000
    assert commands[1, 0] == 12518 - 45000
    assert commands[0, 4] == 60000
    assert commands[1, 4] == 76276 - 45000
    assert np.all(commands[2:] == commands[1])  # Fixed last-target settling.
    gripper = [c[1] for c in sdk.calls if c[0] == "gripper"]
    assert gripper[0] == 35100  # Enable retains the physical opening.
    assert gripper[1:3] == [65100, 5100]
    assert all(c == ("mode", 1, 1, 5, 0) for c in sdk.calls if c[0] == "mode")
    assert_targets_legal(sdk)
    assert result["dispatch_complete"] and result["completed_actions"] == 2
    assert result["target_reached"] is None and result["task_success"] is None
    assert not robot.status()["stopped"] and not robot.status()["connected"]
    assert not any(c[0] == "stop" for c in sdk.calls)


@pytest.mark.parametrize("send_first", [False, True])
def test_hold_and_close_never_command_measured_outside_pose_or_boundary(send_first):
    robot, sdk, _ = make_reference()
    robot.connect()
    if send_first:
        robot.command(legal_target())
    before = copy.deepcopy(sdk.calls)
    np.testing.assert_allclose(robot.hold(), MEASURED_RAW / SDK_SCALE)
    assert sdk.calls == before
    robot.close()
    assert sdk.calls == before + [("disconnect",)]
    assert_targets_legal(sdk)


def test_reference_grasp_hold_never_sends_outside_measured_joint_pose():
    robot, sdk, _ = make_reference()
    robot.connect()
    target = legal_target()
    target[6] = .028
    applied = robot.command(target)
    before = copy.deepcopy(sdk.calls)
    observed = robot.hold_grasp(gripper_target_m=applied[6])
    assert observed[4] == pytest.approx(np.deg2rad(76.276))
    assert sdk.calls == before
    robot.close()
    assert sdk.calls == before + [("disconnect",)]
    assert_targets_legal(sdk)


def test_reference_pre_enable_expected_pose_mismatch_still_never_commands():
    robot, sdk, _ = make_reference()
    expected = MEASURED_RAW / SDK_SCALE
    expected[0] += .1
    with pytest.raises(PiperHardwareError, match="expected_state"):
        robot.connect(expected_state=expected, expected_tolerance=np.full(7, .001))
    robot.close()
    assert sdk.calls == [("connect", False), ("disconnect",)]


@pytest.mark.parametrize("kind", ["joints", "gripper", "arm_status", "motor_status"])
def test_reference_stale_feedback_still_latches_stop_without_target(kind):
    robot, sdk, clock = make_reference()
    robot.connect()
    sdk.calls.clear()
    sdk.fixed_stamps[kind] = clock.wall() - .5
    with pytest.raises(PiperFeedbackError, match="stale"):
        robot.command(legal_target())
    assert sdk.calls == [("stop", 1, 0, 0)]
    robot.close()
    assert sdk.calls[-1] == ("disconnect",)


def test_reference_lost_individual_joint_pair_still_stops_even_with_fresh_cache():
    robot, sdk, clock = make_reference()
    robot.connect()
    sdk.calls.clear()
    sdk.missing_frame_ids.add(0x2A7)
    clock.sleep(.3)
    with pytest.raises(PiperFeedbackError, match="Stale PiPER feedback frame 0x2a7"):
        robot.command(legal_target())
    assert sdk.calls == [("stop", 1, 0, 0)]
    robot.close()


@pytest.mark.parametrize("field,value", [("arm_status", 1), ("err_code", 1),
                                         ("driver_fault", True), ("gripper_fault", True)])
def test_reference_true_hardware_faults_still_stop(field, value):
    robot, sdk, _ = make_reference()
    robot.connect()
    sdk.calls.clear()
    setattr(sdk, field, value)
    with pytest.raises(PiperFeedbackError):
        robot.command(legal_target())
    assert sdk.calls == [("stop", 1, 0, 0)]
    robot.close()


@pytest.mark.parametrize("joint_value", [np.deg2rad(70.1), float("nan")])
def test_reference_illegal_targets_still_never_reach_sdk(joint_value):
    robot, sdk, _ = make_reference()
    robot.connect()
    sdk.calls.clear()
    target = legal_target()
    target[4] = joint_value
    with pytest.raises(ValueError):
        robot.command(target)
    assert not joint_calls(sdk)
    assert robot.status()["stopped"]
    robot.close()


def test_reference_manual_stop_blocks_future_motion_and_cleanup():
    robot, sdk, _ = make_reference()
    robot.connect()
    sdk.calls.clear()
    robot.stop()
    with pytest.raises(PiperHardwareError, match="latched"):
        robot.command(legal_target())
    robot.close()
    assert sdk.calls == [("stop", 1, 0, 0), ("disconnect",)]


def test_reference_cancel_during_enable_sends_no_joint_target():
    robot, sdk, _ = make_reference()
    cancel = threading.Event()
    original = sdk.EnableArm
    def cancel_after_enable(motor):
        original(motor)
        cancel.set()
    sdk.EnableArm = cancel_after_enable
    with pytest.raises(PiperHardwareError, match="cancelled"):
        robot.connect(cancel_event=cancel)
    assert sdk.calls == [("connect", False), ("enable", 7), ("stop", 1, 0, 0), ("disconnect",)]


def test_actual_reference_replay_cancel_stops_before_second_target(tmp_path):
    cancel = threading.Event()
    captured = []
    def configure(sdk):
        captured.append(sdk)
        original = sdk.JointCtrl
        def cancel_after_send(*args):
            original(*args)
            cancel.set()
        sdk.JointCtrl = cancel_after_send
    with pytest.raises(ReplayError, match="cancel"):
        execute_actual_adapter(replay_plan(), tmp_path, configure, cancel=cancel)
    sdk = captured[0]
    assert len(joint_calls(sdk)) == 1
    assert_targets_legal(sdk)
    assert any(c[0] == "stop" for c in sdk.calls)
    assert sdk.calls[-1] == ("disconnect",)
    assert not json.loads((tmp_path / "summary.json").read_text())["success"]


def test_default_projected_policy_still_rejects_outside_feedback_before_enable():
    robot, sdk, _ = make_reference(command_policy="projected")
    with pytest.raises(PiperFeedbackError, match="outside configured"):
        robot.connect()
    assert sdk.calls == [("connect", False), ("disconnect",)]


def test_tracked_plan_still_rejects_outside_initial_state():
    plan = replay_plan()
    plan["replay"]["execution_mode"] = "tracked"
    with pytest.raises(ValueError, match="startup feedback bounds"):
        ReplayPlan.from_mapping(plan)


@pytest.mark.parametrize("execution_mode,command_policy", [("reference", "projected"),
                                                           ("tracked", "reference")])
def test_replay_and_adapter_policies_cannot_silently_disagree(execution_mode, command_policy):
    plan = replay_plan()
    plan["replay"]["execution_mode"] = execution_mode
    plan["robot"]["command_policy"] = command_policy
    with pytest.raises(ValueError, match="requires robot.command_policy"):
        ReplayPlan.from_mapping(plan)


def test_reference_plan_still_rejects_invalid_initial_gripper():
    plan = replay_plan()
    plan["expected_initial_state"][6] = .08
    with pytest.raises(ValueError, match="initial gripper"):
        ReplayPlan.from_mapping(plan)


def test_reference_very_far_outside_feedback_cannot_produce_illegal_delta_target(tmp_path):
    plan = replay_plan()
    plan["expected_initial_state"][4] = np.deg2rad(160)
    captured = []
    def configure(sdk):
        captured.append(sdk)
        sdk.values[4] = 160000
    # The original clip-then-delta rule would produce 115°, which is still
    # outside bounds. Reject that target, never silently send it or clamp away
    # the 45° cap. This extreme case is not the present 76.276° feedback.
    with pytest.raises(ReplayError, match="outside configured"):
        execute_actual_adapter(plan, tmp_path, configure)
    assert not joint_calls(captured[0])
    assert any(c[0] == "stop" for c in captured[0].calls)
