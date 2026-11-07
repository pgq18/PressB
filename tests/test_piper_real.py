"""Hardware-free PiPER tests exercise physical units and fail-closed paths."""
from types import SimpleNamespace as NS
import threading

import numpy as np
import pytest

from pressb.robots.base import RobotHardware
from pressb.robots.piper.real import (
    PiperFeedbackError,
    PiperHardware,
    PiperHardwareConfig,
    PiperHardwareError,
    check_can_interface,
)


class Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def wall(self):
        return 1_800_000_000.0 + self.now

    def sleep(self, duration):
        self.now += duration


class SDK:
    def __init__(self, clock):
        self.clock = clock
        self.connected = False
        self.calls = []
        self.values = np.array([10000, 20000, -30000, 40000, -50000, 60000, 40000])
        self.fixed_stamps = {}
        self.motor_enabled = [False] * 6
        self.gripper_enabled = False
        self.enable_succeeds = True
        self.arm_status = 0
        self.err_code = 0
        self.driver_fault = False
        self.gripper_fault = False
        self.gripper_effort = 0
        self.gripper_status_extra = 0
        self.fail_joint_send = False
        self.fail_transmit_ids = set()
        self.missing_frame_ids = set()
        self.bus = NS(CAN_STATUS=NS(SEND_MESSAGE_SUCCESS=0), SendCanMessage=self.raw_send,
                      callback_function=self.raw_receive)

    def raw_receive(self, message):
        pass

    def emit(self, *ids):
        for can_id in ids:
            if can_id not in self.missing_frame_ids:
                payload = bytes(8)
                if can_id == 0x2A8:
                    payload = (int(self.values[6]).to_bytes(4, "big", signed=True)
                               + int(self.gripper_effort).to_bytes(2, "big", signed=True)
                               + bytes([self.gripper_status(), 0]))
                self.bus.callback_function(NS(arbitration_id=can_id, timestamp=self.clock.wall(),
                                              dlc=8, data=payload))

    def gripper_status(self):
        return (int(self.gripper_enabled) << 6) | (int(self.gripper_fault) << 4) | self.gripper_status_extra

    def raw_send(self, can_id, data):
        return 1 if can_id in self.fail_transmit_ids else 0

    def GetCanBus(self):
        return self.bus

    def stamp(self, kind):
        return self.fixed_stamps.get(kind, self.clock.wall())

    def get_connect_status(self):
        return self.connected

    def ConnectPort(self, *, piper_init):
        self.calls.append(("connect", piper_init))
        self.connected = True

    def DisconnectPort(self):
        self.calls.append(("disconnect",))
        self.connected = False

    def GetArmJointMsgs(self):
        self.emit(0x2A5, 0x2A6, 0x2A7)
        return NS(time_stamp=self.stamp("joints"), joint_state=NS(
            **{f"joint_{i + 1}": value for i, value in enumerate(self.values[:6])}))

    def GetArmGripperMsgs(self):
        self.emit(0x2A8)
        return NS(time_stamp=self.stamp("gripper"), gripper_state=NS(
            grippers_angle=self.values[6], grippers_effort=self.gripper_effort,
            status_code=self.gripper_status(), foc_status=NS(driver_enable_status=self.gripper_enabled,
                sensor_status=self.gripper_fault, voltage_too_low=False, motor_overheating=False,
                driver_overcurrent=False, driver_overheating=False, driver_error_status=False)))

    def GetArmStatus(self):
        self.emit(0x2A1)
        return NS(time_stamp=self.stamp("arm_status"),
                  arm_status=NS(arm_status=self.arm_status, err_code=self.err_code))

    def GetArmLowSpdInfoMsgs(self):
        self.emit(*range(0x261, 0x267))
        return NS(time_stamp=self.stamp("motor_status"), **{
            f"motor_{i + 1}": NS(foc_status=NS(driver_enable_status=enabled,
                collision_status=self.driver_fault, voltage_too_low=False, motor_overheating=False,
                driver_overcurrent=False, driver_overheating=False, driver_error_status=False,
                stall_status=False))
            for i, enabled in enumerate(self.motor_enabled)})

    def EnableArm(self, motor):
        self.calls.append(("enable", motor))
        self.bus.SendCanMessage(0x471, [])
        if self.enable_succeeds:
            self.motor_enabled = [True] * 6

    def GripperCtrl(self, *values):
        self.calls.append(("gripper", *values))
        self.bus.SendCanMessage(0x159, [])
        if self.enable_succeeds:
            self.gripper_enabled = True

    def MotionCtrl_2(self, *values):
        self.calls.append(("mode", *values))
        self.bus.SendCanMessage(0x151, [])

    def JointCtrl(self, *values):
        self.calls.append(("joints", *values))
        for can_id in (0x155, 0x156, 0x157):
            self.bus.SendCanMessage(can_id, [])
        if self.fail_joint_send:
            raise OSError("CAN transmit failed")

    def MotionCtrl_1(self, *values):
        self.calls.append(("stop", *values))
        self.bus.SendCanMessage(0x150, [values[0], *([0] * 7)])


def make_hardware(**config):
    clock = Clock()
    sdk = SDK(clock)
    hardware = PiperHardware(config, sdk_factory=lambda name: sdk, can_probe=lambda name: None,
                             monotonic=clock.monotonic, wall_time=clock.wall, sleep=clock.sleep)
    return hardware, sdk, clock


def test_read_only_connect_requires_live_feedback_and_never_moves():
    hardware, sdk, _ = make_hardware()
    hardware.connect()
    assert isinstance(hardware, RobotHardware)
    state = hardware.read_state()
    np.testing.assert_allclose(state[:6], np.deg2rad([10, 20, -30, 40, -50, 60]))
    assert state[6] == .04  # Full width, not per-finger displacement or centimetres.
    state[:] = 999
    assert hardware.read_state()[6] == .04
    with pytest.raises(PiperHardwareError, match="actuation is disabled"):
        hardware.command(hardware.read_state())
    hardware.hold()
    hardware.stop()
    hardware.close()
    assert sdk.calls == [("connect", False), ("disconnect",)]


def test_can_missing_fails_before_constructing_sdk():
    created = []
    hardware = PiperHardware({"can_name": "pressb-missing-can"},
                             sdk_factory=lambda name: created.append(name))
    with pytest.raises(PiperHardwareError, match="unavailable"):
        hardware.connect()
    assert created == []
    assert not hardware.status()["connected"]


def test_non_can_network_interface_is_rejected():
    with pytest.raises(PiperHardwareError, match="not a CAN"):
        check_can_interface("lo")


def test_existing_sdk_connection_is_not_adopted_or_disconnected():
    hardware, sdk, _ = make_hardware()
    sdk.connected = True
    with pytest.raises(PiperHardwareError, match="another controller"):
        hardware.connect()
    assert sdk.calls == []
    assert sdk.connected


def test_actuation_requires_a_checked_transport_before_enable():
    hardware, sdk, _ = make_hardware(actuation_enabled=True)
    sdk.GetCanBus = lambda: NS()
    with pytest.raises(PiperHardwareError, match="checked GetCanBus"):
        hardware.connect()
    assert sdk.calls == [("disconnect",)]


@pytest.mark.parametrize("stamp", [0, -1, float("nan"), float("inf"), 1, 9e20])
def test_invalid_initial_timestamps_never_enable(stamp):
    hardware, sdk, _ = make_hardware(actuation_enabled=True, connect_timeout_s=.03)
    sdk.fixed_stamps["joints"] = stamp
    with pytest.raises(PiperFeedbackError, match="live PiPER feedback"):
        hardware.connect()
    assert sdk.calls == [("connect", False), ("disconnect",)]


def test_initial_feedback_must_advance_even_if_timestamp_is_recent():
    hardware, sdk, clock = make_hardware(actuation_enabled=True, connect_timeout_s=.03)
    sdk.fixed_stamps["joints"] = clock.wall()
    sdk.fixed_stamps["gripper"] = clock.wall()
    with pytest.raises(PiperFeedbackError, match="not advancing"):
        hardware.connect()
    assert not any(call[0] == "enable" for call in sdk.calls)


def test_enable_preserves_measured_gripper_and_verifies_all_drivers():
    hardware, sdk, _ = make_hardware(actuation_enabled=True)
    hardware.connect()
    assert ("enable", 7) in sdk.calls
    assert ("gripper", 40000, 1000, 1, 0) in sdk.calls
    assert not any(call[0] in ("joints", "mode", "stop") for call in sdk.calls)
    assert hardware.status()["motors_enabled"]
    sdk.motor_enabled[4] = False
    with pytest.raises(PiperFeedbackError, match="enable feedback was lost"):
        hardware.read_state()
    assert ("stop", 1, 0, 0) in sdk.calls


def test_enable_timeout_emergency_stops_partial_enable_and_disconnects():
    hardware, sdk, _ = make_hardware(actuation_enabled=True, enable_timeout_s=.03)
    sdk.enable_succeeds = False
    with pytest.raises(PiperHardwareError, match="Timed out.*enable"):
        hardware.connect()
    assert sdk.calls[-2:] == [("stop", 1, 0, 0), ("disconnect",)]
    assert not hardware.status()["connected"]


def test_command_uses_millidegrees_and_full_width_micrometres():
    hardware, sdk, _ = make_hardware(actuation_enabled=True, max_joint_velocity_rad_s=[10.] * 6,
                                    max_gripper_velocity_m_s=1.)
    hardware.connect()
    target = np.r_[np.deg2rad([11, 21, -31, 41, -51, 61]), .045]
    result = hardware.command(target)
    np.testing.assert_allclose(result, target)
    assert sdk.calls[-3:] == [("mode", 1, 1, 10, 0),
                             ("joints", 11000, 21000, -31000, 41000, -51000, 61000),
                             ("gripper", 45000, 1000, 1, 0)]
    result[:] = 999
    assert hardware._last_command[6] == .045


def test_rate_limit_uses_measured_start_and_elapsed_time_without_catchup():
    hardware, sdk, clock = make_hardware(actuation_enabled=True)
    hardware.connect()
    initial = hardware.read_state()
    target = initial + np.r_[[.2] * 6, .02]
    first = hardware.command(target)
    velocity = np.r_[[.5] * 6, .02]
    assert np.all(first >= initial)
    assert np.all(first - initial <= velocity / 30 + 1e-12)
    same_tick = hardware.command(target)
    np.testing.assert_array_equal(same_tick, first)
    clock.sleep(.01)
    next_tick = hardware.command(target)
    assert np.all(next_tick - same_tick <= velocity * .01 + 1e-12)
    clock.sleep(10)
    after_pause = hardware.command(target)
    assert np.all(after_pause - next_tick <= velocity / 30 + 1e-12)
    assert sdk.calls[-1][0] == "gripper"


@pytest.mark.parametrize("target", [np.zeros(6), np.zeros((1, 7)), [0] * 6 + [-.001],
                                   [0] * 6 + [.071], [10] * 7, [float("nan")] * 7,
                                   [float("inf")] * 7, [1j] * 7, ["0"] * 7])
def test_invalid_target_latches_stop_before_sending_joint_motion(target):
    hardware, sdk, _ = make_hardware(actuation_enabled=True)
    hardware.connect()
    sdk.calls.clear()
    with pytest.raises(ValueError):
        hardware.command(target)
    assert sdk.calls == [("stop", 1, 0, 0)]
    with pytest.raises(PiperHardwareError, match="latched"):
        hardware.command(hardware.read_state())


@pytest.mark.parametrize("bound", ["lower", "upper"])
def test_float32_metadata_bounds_are_accepted_and_clamped_to_exact_limits(bound):
    hardware, _, _ = make_hardware(actuation_enabled=True)
    target = np.array(hardware.describe()["action"][bound], dtype=np.float32)
    validated = hardware.validate_target(target)
    assert np.all(validated >= hardware._lower)
    assert np.all(validated <= hardware._upper)
    expected = np.clip(target.astype(np.float64), hardware._lower, hardware._upper)
    np.testing.assert_array_equal(validated, expected)
    hardware.connect()
    hardware.command(target)
    assert not hardware.status()["stopped"]


@pytest.mark.parametrize("bound,delta", [("lower", -2e-6), ("upper", 2e-6)])
def test_targets_beyond_abi_quantization_tolerance_remain_rejected(bound, delta):
    hardware, _, _ = make_hardware()
    target = np.array(hardware.describe()["action"][bound])
    target[2] += delta
    with pytest.raises(ValueError, match="outside configured"):
        hardware.validate_target(target)


@pytest.mark.parametrize("opcode", [0, 2])
def test_stop_latch_does_not_allow_reset_or_other_motion_control_opcodes(opcode):
    hardware, sdk, _ = make_hardware(actuation_enabled=True)
    hardware.connect()
    hardware.stop()
    with pytest.raises(PiperHardwareError, match="stop requested"):
        sdk.GetCanBus().SendCanMessage(0x150, [opcode, *([0] * 7)])


@pytest.mark.parametrize("kind", ["joints", "gripper", "arm_status", "motor_status"])
def test_stale_feedback_latches_stop_and_sends_no_new_target(kind):
    hardware, sdk, clock = make_hardware(actuation_enabled=True)
    hardware.connect()
    target = hardware.read_state()
    sdk.calls.clear()
    sdk.fixed_stamps[kind] = clock.wall() - .5
    with pytest.raises(PiperFeedbackError, match="stale"):
        hardware.command(target)
    assert sdk.calls == [("stop", 1, 0, 0)]


def test_stop_latches_without_automatic_reset_or_redundant_stop_messages():
    hardware, sdk, _ = make_hardware(actuation_enabled=True)
    hardware.connect()
    sdk.calls.clear()
    hardware.stop()
    hardware.stop()
    with pytest.raises(PiperHardwareError, match="latched"):
        hardware.hold()
    hardware.close()
    assert sdk.calls == [("stop", 1, 0, 0), ("disconnect",)]


def test_normal_hold_and_close_use_actual_pose_without_disable_or_home():
    hardware, sdk, _ = make_hardware(actuation_enabled=True)
    hardware.connect()
    sdk.values[0] = 12000
    sdk.values[6] = 37000
    held = hardware.hold()
    assert held[0] == pytest.approx(np.deg2rad(12))
    assert held[6] == .037
    assert not hardware.status()["stopped"]
    hardware.close()
    hardware.close()
    assert sdk.calls[-4:] == [("mode", 1, 1, 10, 0),
                             ("joints", 12000, 20000, -30000, 40000, -50000, 60000),
                             ("gripper", 37000, 1000, 1, 0), ("disconnect",)]
    assert not any(call[0] == "stop" for call in sdk.calls)


def test_close_after_feedback_loss_requests_stop_and_still_disconnects():
    hardware, sdk, clock = make_hardware(actuation_enabled=True)
    hardware.connect()
    sdk.fixed_stamps["joints"] = clock.wall() - 1
    with pytest.raises(PiperHardwareError, match="failed to hold"):
        hardware.close()
    assert sdk.calls[-2:] == [("stop", 1, 0, 0), ("disconnect",)]


def test_failed_joint_transmit_latches_stop_before_gripper_motion():
    hardware, sdk, _ = make_hardware(actuation_enabled=True)
    hardware.connect()
    sdk.fail_joint_send = True
    sdk.calls.clear()
    with pytest.raises(OSError, match="transmit"):
        hardware.command(hardware.read_state())
    assert [call[0] for call in sdk.calls] == ["mode", "joints", "stop"]


@pytest.mark.parametrize("can_id", [0x151, 0x155, 0x156, 0x157, 0x159])
def test_sdk_log_only_transport_failure_does_not_record_unsent_setpoint(can_id):
    hardware, sdk, _ = make_hardware(actuation_enabled=True)
    hardware.connect()
    sdk.fail_transmit_ids.add(can_id)
    with pytest.raises(PiperHardwareError, match="CAN transmission failed"):
        hardware.command(hardware.read_state())
    assert hardware._last_command is None
    assert hardware.status()["stopped"]
    assert hardware.status()["emergency_stop_sent"]


def test_failed_stop_send_is_reported_and_can_be_retried():
    hardware, sdk, _ = make_hardware(actuation_enabled=True)
    hardware.connect()
    sdk.fail_transmit_ids.add(0x150)
    with pytest.raises(PiperHardwareError, match="emergency stop send failed"):
        hardware.stop()
    assert hardware.status()["stopped"]
    assert hardware.status()["emergency_stop_sent"] is False
    sdk.fail_transmit_ids.clear()
    hardware.stop()
    assert hardware.status()["emergency_stop_sent"] is True
    hardware.close()
    assert sdk.bus.SendCanMessage == sdk.raw_send
    assert sdk.bus.callback_function == sdk.raw_receive


@pytest.mark.parametrize("blocked_id", [0x156, 0x159])
def test_stop_latches_during_blocked_send_and_rejects_remaining_frames_and_late_success(blocked_id):
    hardware, sdk, _ = make_hardware(actuation_enabled=True)
    entered, release = threading.Event(), threading.Event()
    enabled = threading.Event()
    transmitted = []
    original_send = sdk.bus.SendCanMessage

    def blocked_send(can_id, data):
        if enabled.is_set() and can_id == blocked_id:
            entered.set()
            if not release.wait(timeout=2):
                raise TimeoutError("Test did not release blocked CAN send")
        result = original_send(can_id, data)
        if enabled.is_set():
            transmitted.append(can_id)
        return result

    sdk.bus.SendCanMessage = blocked_send
    hardware.connect()
    target = hardware.read_state()
    enabled.set()
    command_errors, stop_errors, results = [], [], []

    def command():
        try:
            results.append(hardware.command(target))
        except Exception as error:
            command_errors.append(error)

    def stop():
        try:
            hardware.stop()
        except Exception as error:
            stop_errors.append(error)

    worker = threading.Thread(target=command)
    stopper = threading.Thread(target=stop)
    worker.start()
    try:
        assert entered.wait(timeout=1)
        stopper.start()
        assert hardware._stop_requested.wait(timeout=1)
        assert stopper.is_alive()  # No claim that blocked transmission can be cancelled.
        assert 0x150 not in transmitted
    finally:
        release.set()
        worker.join(timeout=2)
        if stopper.ident is not None:
            stopper.join(timeout=2)
    assert not worker.is_alive() and not stopper.is_alive()
    assert not results and not stop_errors
    assert len(command_errors) == 1
    assert isinstance(command_errors[0], PiperHardwareError)
    assert "stop requested" in str(command_errors[0])
    assert hardware._last_command is None
    expected = [0x151, 0x155, 0x156]
    if blocked_id == 0x159:
        expected += [0x157, 0x159]
    assert transmitted == expected + [0x150]
    assert hardware.status()["stop_requested"]
    assert hardware.status()["emergency_stop_sent"]


@pytest.mark.parametrize("can_id", [0x2A5, 0x2A6, 0x2A7, 0x2A8, 0x2A1, 0x263])
def test_missing_individual_feedback_frame_blocks_startup_even_with_live_sdk_cache(can_id):
    hardware, sdk, _ = make_hardware(actuation_enabled=True, connect_timeout_s=.03)
    sdk.missing_frame_ids.add(can_id)
    with pytest.raises(PiperFeedbackError, match="Missing PiPER feedback frame"):
        hardware.connect()
    assert sdk.calls == [("connect", False), ("disconnect",)]


def test_stalled_individual_joint_pair_cannot_be_hidden_by_aggregate_timestamp():
    hardware, sdk, clock = make_hardware(actuation_enabled=True)
    hardware.connect()
    sdk.missing_frame_ids.add(0x2A6)
    clock.sleep(.3)
    with pytest.raises(PiperFeedbackError, match="Stale PiPER feedback frame 0x2a6"):
        hardware.read_state()
    assert hardware.status()["stopped"]


@pytest.mark.parametrize("fault", ["stale_motor", "motor", "gripper"])
def test_initial_driver_status_is_validated_before_any_actuation(fault):
    hardware, sdk, clock = make_hardware(actuation_enabled=True, connect_timeout_s=.03)
    if fault == "stale_motor":
        sdk.fixed_stamps["motor_status"] = clock.wall() - 1
    elif fault == "motor":
        sdk.driver_fault = True
    else:
        sdk.gripper_fault = True
    with pytest.raises(PiperFeedbackError):
        hardware.connect()
    assert sdk.calls == [("connect", False), ("disconnect",)]


@pytest.mark.parametrize("field,value", [("arm_status", 7), ("err_code", 1),
                                        ("driver_fault", True), ("gripper_fault", True)])
def test_arm_and_driver_faults_latch_stop(field, value):
    hardware, sdk, _ = make_hardware(actuation_enabled=True)
    hardware.connect()
    setattr(sdk, field, value)
    with pytest.raises(PiperFeedbackError):
        hardware.read_state()
    assert hardware.status()["stopped"]


@pytest.mark.parametrize("config", [
    {"actuation_enabled": "true"}, {"actuation_enabled": 1}, {"control_hz": 0},
    {"control_hz": float("nan")}, {"feedback_timeout_s": float("inf")},
    {"joint_lower_rad": [0] * 5}, {"joint_upper_rad": [float("inf")] * 6},
    {"joint_lower_rad": [4.] * 6}, {"max_joint_velocity_rad_s": [0.] * 6},
    {"max_gripper_velocity_m_s": -1}, {"gripper_min_width_m": -.001},
    {"gripper_max_width_m": .2}, {"gripper_max_width_m": float("nan")},
    {"speed_percent": True}, {"speed_percent": 101}, {"gripper_effort": 5001},
    {"can_name": "../can0"},
])
def test_configuration_requires_explicit_finite_limits(config):
    with pytest.raises(ValueError):
        PiperHardwareConfig(**config)


def test_describe_exposes_si_bounds_and_feedback_limitations_without_connecting():
    hardware, sdk, _ = make_hardware()
    description = hardware.describe()
    assert description["num_envs"] == 1
    assert description["robot_type"] == "piper"
    assert description["state"]["units"] == ["rad"] * 6 + ["m"]
    assert description["action"]["upper"][-1] == .07
    assert description["automatic_homing"] is False
    assert description["actuation_enabled"] is False
    assert "Individual joint-pair" in description["feedback_freshness"]
    assert sdk.calls == []


def startup_hardware(**config):
    """The passive real-arm sample, reproduced without loading the real SDK."""
    config.setdefault("actuation_enabled", True)
    config.setdefault("joint_startup_recovery_tolerance_rad", np.deg2rad([0, 3, 3, 0, 0, 0]))
    hardware, sdk, clock = make_hardware(**config)
    sdk.values = np.array([667, -1507, 2381, 4218, 21069, -2540, 0])
    return hardware, sdk, clock


def test_startup_recovery_is_disabled_by_default_and_never_enables_outside_limits():
    hardware, sdk, _ = startup_hardware(joint_startup_recovery_tolerance_rad=(0.,) * 6)
    with pytest.raises(PiperFeedbackError, match="outside configured"):
        hardware.connect()
    assert sdk.calls == [("connect", False), ("disconnect",)]


def test_startup_recovery_preserves_measurements_and_nominal_action_limits():
    hardware, sdk, _ = startup_hardware()
    nominal, _, _ = make_hardware()
    assert hardware.describe()["action"] == nominal.describe()["action"]
    hardware.connect()
    np.testing.assert_allclose(hardware.read_state()[:6],
                               np.deg2rad([.667, -1.507, 2.381, 4.218, 21.069, -2.54]))
    assert hardware.status()["startup_recovery"]["active_joint_indices"] == [1, 2]
    assert not any(call[0] in ("joints", "mode") for call in sdk.calls)
    with pytest.raises(ValueError, match="outside configured"):
        hardware.validate_target(hardware.read_state())


def test_startup_recovery_rate_limits_quantized_inward_steps_without_boundary_jump():
    hardware, sdk, clock = startup_hardware()
    hardware.connect()
    initial = previous = hardware.read_state()
    target = initial.copy()
    target[1:3] = [.1, -.1]
    for index in range(12):
        clock.sleep(1 / 30)
        applied = hardware.command(target)
        assert np.all(np.abs(applied - previous) <= np.r_[[.5 / 30] * 6, .02 / 30] + 1e-12)
        assert applied[1] >= previous[1] and applied[2] <= previous[2]
        if index == 0:
            assert initial[1] < applied[1] < 0 < applied[2] < initial[2]
        sdk.values = np.rint(applied * hardware._sdk_scale).astype(int)
        previous = applied
    hardware.read_state()
    assert hardware.status()["startup_recovery"]["active_joint_indices"] == []
    assert np.all(np.abs(previous - target) <= 1 / hardware._sdk_scale)


def test_startup_hold_does_not_reverse_setpoint_toward_lagging_measurement():
    hardware, sdk, clock = startup_hardware()
    hardware.connect()
    target = hardware.read_state()
    target[1:3] = [.1, -.1]
    previous = hardware.command(target)
    # The fake physical arm deliberately remains at the initial measured pose.
    clock.sleep(1 / 30)
    held = hardware.hold()
    np.testing.assert_array_equal(held, previous)
    assert sdk.calls[-1][0] == "gripper"


def test_startup_hold_rate_limits_measurement_ahead_of_last_setpoint():
    hardware, sdk, clock = startup_hardware(max_joint_velocity_rad_s=(.05,) * 6)
    hardware.connect()
    first = hardware.hold()
    sdk.values[1] += 50
    sdk.values[2] -= 50
    clock.sleep(1 / 30)
    second = hardware.hold()
    assert np.all(np.abs(second[:6] - first[:6]) <= .05 / 30 + 1e-12)
    assert second[1] == sdk.values[1] / hardware._sdk_scale[1] > first[1]


@pytest.mark.parametrize("operation", ["command", "hold"])
@pytest.mark.parametrize("entered_nominal", [False, True])
def test_startup_measurement_overtaking_rate_limit_stops_without_outward_reference(operation, entered_nominal):
    hardware, sdk, clock = startup_hardware(max_joint_velocity_rad_s=(.01,) * 6)
    hardware.connect()
    hardware.hold()
    sdk.values[1] = 0 if entered_nominal else -500
    clock.sleep(1 / 30)
    before = len([call for call in sdk.calls if call[0] == "joints"])
    with pytest.raises(PiperHardwareError, match="projected setpoint"):
        if operation == "hold":
            hardware.hold()
        else:
            target = sdk.values / hardware._sdk_scale
            target[1:3] = [.1, -.1]
            hardware.command(target)
    assert len([call for call in sdk.calls if call[0] == "joints"]) == before
    assert sdk.calls[-1] == ("stop", 1, 0, 0)


def test_read_then_hold_cannot_jump_from_retired_outside_setpoint():
    hardware, sdk, clock = startup_hardware(max_joint_velocity_rad_s=(.01,) * 6)
    hardware.connect()
    hardware.hold()
    sdk.values[1:3] = [0, 0]
    hardware.read_state()
    assert hardware.status()["startup_recovery"]["active_joint_indices"] == []
    clock.sleep(1 / 30)
    previous_count = len([call for call in sdk.calls if call[0] == "joints"])
    with pytest.raises(PiperHardwareError, match="projected setpoint"):
        hardware.hold()
    assert len([call for call in sdk.calls if call[0] == "joints"]) == previous_count
    assert sdk.calls[-1] == ("stop", 1, 0, 0)


@pytest.mark.parametrize("inward_progress", [0, 500])
def test_startup_feedback_slack_is_tiny_and_does_not_ratchet_outward(inward_progress):
    hardware, sdk, _ = startup_hardware()
    hardware.connect()
    sdk.values[1] += inward_progress
    hardware.read_state()
    sdk.values[1] -= 2
    hardware.read_state()  # Exactly two encoder units of feedback noise.
    sdk.values[1] -= 1
    with pytest.raises(PiperFeedbackError, match="moved outward"):
        hardware.read_state()
    assert hardware.status()["stopped"]
    assert sdk.calls[-1] == ("stop", 1, 0, 0)


def test_configured_feedback_slack_accepts_observed_startup_settling_only():
    hardware, sdk, _ = startup_hardware(
        startup_recovery_feedback_slack_rad=np.deg2rad([0, .25, .25, 0, 0, 0]))
    hardware.connect()
    initial = hardware.read_state()
    initial_best = hardware.status()["startup_recovery"]["best_excursion_rad"]
    sdk.values[1] -= 39
    sdk.values[2] += 132  # The observed 0.132-degree startup response.
    measured = hardware.read_state()
    np.testing.assert_allclose(np.rad2deg(measured[1:3] - initial[1:3]), [-.039, .132])
    assert hardware.status()["startup_recovery"]["best_excursion_rad"] == initial_best
    assert not hardware.status()["stopped"]
    assert not any(call[0] == "joints" for call in sdk.calls)


def test_configured_feedback_slack_is_fixed_and_cannot_ratchet_outward():
    hardware, sdk, _ = startup_hardware(
        startup_recovery_feedback_slack_rad=np.deg2rad([0, .25, .25, 0, 0, 0]))
    hardware.connect()
    initial_raw = sdk.values[2]
    for displacement in [100, 200, 250]:
        sdk.values[2] = initial_raw + displacement
        hardware.read_state()
    sdk.values[2] = initial_raw + 251
    with pytest.raises(PiperFeedbackError, match="moved outward"):
        hardware.read_state()
    assert hardware.status()["stopped"]


def test_configured_feedback_slack_still_shrinks_after_inward_progress():
    hardware, sdk, _ = startup_hardware(
        startup_recovery_feedback_slack_rad=np.deg2rad([0, .25, .25, 0, 0, 0]))
    hardware.connect()
    sdk.values[2] -= 200
    hardware.read_state()
    sdk.values[2] += 250
    hardware.read_state()
    sdk.values[2] += 1  # Within initial+slack, but outside best+slack.
    with pytest.raises(PiperFeedbackError, match="moved outward"):
        hardware.read_state()


@pytest.mark.parametrize("tolerance_deg", [2.5, 3.])
def test_feedback_slack_cannot_exceed_total_configured_startup_tolerance(tolerance_deg):
    hardware, sdk, _ = startup_hardware(
        joint_startup_recovery_tolerance_rad=np.deg2rad([0, 3, tolerance_deg, 0, 0, 0]),
        startup_recovery_feedback_slack_rad=np.deg2rad([0, .25, .25, 0, 0, 0]))
    sdk.values[2] = round(tolerance_deg * 1000) - 50
    hardware.connect()
    sdk.values[2] += 50
    hardware.read_state()
    sdk.values[2] += 1
    with pytest.raises(PiperFeedbackError, match="moved outward"):
        hardware.read_state()


@pytest.mark.parametrize("operation", ["command", "hold"])
def test_feedback_slack_does_not_extend_sent_startup_reference_envelope(operation):
    hardware, sdk, _ = startup_hardware(
        max_joint_velocity_rad_s=(.01,) * 6,
        startup_recovery_feedback_slack_rad=np.deg2rad([0, .25, .25, 0, 0, 0]))
    hardware.connect()
    sdk.values[2] += 132
    hardware.read_state()  # Measured settling is allowed, command expansion is not.
    with pytest.raises(PiperHardwareError, match="projected setpoint"):
        if operation == "hold":
            hardware.hold()
        else:
            target = sdk.values / hardware._sdk_scale
            target[1:3] = [.1, -.1]
            hardware.command(target)
    assert not any(call[0] == "joints" for call in sdk.calls)
    assert hardware.status()["stopped"]


@pytest.mark.parametrize("slack", [
    [.001] * 5, [-.001] * 6, [float("nan")] * 6,
    np.deg2rad([.251] * 6), .001, [True] * 6,
])
def test_recovery_feedback_slack_requires_finite_per_joint_values_with_quarter_degree_cap(slack):
    with pytest.raises(ValueError):
        PiperHardwareConfig(startup_recovery_feedback_slack_rad=slack)


def test_startup_recovery_allowance_clears_per_joint_and_cannot_be_reopened():
    hardware, sdk, _ = startup_hardware()
    hardware.connect()
    sdk.values[1] = 0
    hardware.read_state()
    assert hardware.status()["startup_recovery"]["active_joint_indices"] == [2]
    sdk.values[1] = -1
    with pytest.raises(PiperFeedbackError, match="outside configured"):
        hardware.read_state()
    assert hardware.status()["stopped"]


def test_startup_recovery_does_not_authorize_a_new_joint_excursion():
    hardware, sdk, _ = startup_hardware()
    hardware.connect()
    sdk.values[0] = 154001
    with pytest.raises(PiperFeedbackError, match="outside configured"):
        hardware.read_state()


def test_startup_recovery_exceeding_per_joint_tolerance_never_enables():
    hardware, sdk, _ = startup_hardware()
    sdk.values[2] = 3001
    with pytest.raises(PiperFeedbackError, match="outside configured"):
        hardware.connect()
    assert sdk.calls == [("connect", False), ("disconnect",)]


def test_startup_recovery_deadline_latches_stop():
    hardware, sdk, clock = startup_hardware(startup_recovery_timeout_s=.1)
    hardware.connect()
    clock.sleep(.101)
    with pytest.raises(PiperFeedbackError, match="recovery timed out"):
        hardware.hold()
    assert sdk.calls[-1] == ("stop", 1, 0, 0)
    hardware.close()
    assert sdk.calls[-1] == ("disconnect",)


@pytest.mark.parametrize("config", [
    {"joint_startup_recovery_tolerance_rad": [.01] * 5},
    {"joint_startup_recovery_tolerance_rad": [-.01] * 6},
    {"joint_startup_recovery_tolerance_rad": [float("nan")] * 6},
    {"joint_startup_recovery_tolerance_rad": np.deg2rad([3.001] * 6)},
    {"joint_startup_recovery_tolerance_rad": .01},
    {"startup_recovery_timeout_s": 0},
    {"startup_recovery_timeout_s": float("inf")},
])
def test_startup_recovery_config_requires_small_finite_per_joint_allowances(config):
    with pytest.raises(ValueError):
        PiperHardwareConfig(**config)


@pytest.mark.parametrize("kwargs", [
    {"expected_state": [0.] * 7},
    {"expected_tolerance": [1.] * 7},
    {"expected_state": [0.] * 6, "expected_tolerance": [1.] * 7},
    {"expected_state": [float("nan")] * 7, "expected_tolerance": [1.] * 7},
    {"expected_state": [0.] * 7, "expected_tolerance": [0.] * 7},
    {"expected_state": [0.] * 7, "expected_tolerance": [-1.] * 7},
    {"expected_state": [0.] * 7, "expected_tolerance": [float("inf")] * 7},
])
def test_expected_start_arguments_are_validated_before_can_access(kwargs):
    hardware, sdk, _ = make_hardware(actuation_enabled=True)
    probed = []
    hardware._can_probe = probed.append
    with pytest.raises(ValueError):
        hardware.connect(**kwargs)
    assert probed == [] and sdk.calls == []


def test_expected_start_mismatch_disconnects_without_any_enable_or_stop():
    hardware, sdk, _ = startup_hardware()
    expected = sdk.values / hardware._sdk_scale
    expected[0] += .1
    with pytest.raises(PiperHardwareError, match="reviewed expected_state"):
        hardware.connect(expected_state=expected, expected_tolerance=np.full(7, .001))
    assert sdk.calls == [("connect", False), ("disconnect",)]


def test_expected_start_accepts_reviewed_outside_pose_before_recovery_enable():
    hardware, sdk, _ = startup_hardware()
    hardware.connect(expected_state=sdk.values / hardware._sdk_scale,
                     expected_tolerance=np.r_[np.full(6, np.deg2rad(1)), .001])
    assert hardware.status()["motors_enabled"]
    assert hardware.status()["startup_recovery"]["active_joint_indices"] == [1, 2]


def test_expected_start_is_rechecked_immediately_before_enable():
    hardware, sdk, _ = make_hardware(actuation_enabled=True)
    expected = sdk.values / hardware._sdk_scale
    original = hardware._await_feedback

    def move_after_feedback(**kwargs):
        original(**kwargs)
        sdk.values[0] += 10000

    hardware._await_feedback = move_after_feedback
    with pytest.raises(PiperHardwareError, match="reviewed expected_state"):
        hardware.connect(expected_state=expected, expected_tolerance=np.full(7, .001))
    assert sdk.calls == [("connect", False), ("disconnect",)]


def test_connect_cancellation_and_bad_event_fail_before_can_access():
    hardware, sdk, _ = make_hardware(actuation_enabled=True)
    probed = []
    hardware._can_probe = probed.append
    with pytest.raises(ValueError, match="is_set"):
        hardware.connect(cancel_event=True)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(PiperHardwareError, match="cancelled"):
        hardware.connect(cancel_event=cancel)
    assert probed == [] and sdk.calls == []


def test_connect_cancellation_during_feedback_wait_never_enables():
    hardware, sdk, _ = make_hardware(actuation_enabled=True)
    cancel = threading.Event()
    original = sdk.GetArmJointMsgs

    def cancel_on_feedback():
        cancel.set()
        return original()

    sdk.GetArmJointMsgs = cancel_on_feedback
    with pytest.raises(PiperHardwareError, match="cancelled"):
        hardware.connect(cancel_event=cancel)
    assert sdk.calls == [("connect", False), ("disconnect",)]


def test_connect_rechecks_cancellation_after_fresh_feedback_before_enable():
    hardware, sdk, _ = make_hardware(actuation_enabled=True)
    cancel = threading.Event()
    original = hardware._await_feedback

    def cancel_after_feedback(**kwargs):
        original(**kwargs)
        cancel.set()

    hardware._await_feedback = cancel_after_feedback
    with pytest.raises(PiperHardwareError, match="cancelled"):
        hardware.connect(cancel_event=cancel)
    assert sdk.calls == [("connect", False), ("disconnect",)]


def test_connect_cancellation_during_enable_stops_and_skips_gripper():
    hardware, sdk, _ = make_hardware(actuation_enabled=True)
    cancel = threading.Event()
    original = sdk.EnableArm

    def cancel_after_arm_enable(motor):
        original(motor)
        cancel.set()

    sdk.EnableArm = cancel_after_arm_enable
    with pytest.raises(PiperHardwareError, match="cancelled"):
        hardware.connect(cancel_event=cancel)
    assert sdk.calls == [("connect", False), ("enable", 7), ("stop", 1, 0, 0), ("disconnect",)]


def test_connect_cancellation_during_gripper_enable_cannot_return_success():
    hardware, sdk, _ = make_hardware(actuation_enabled=True)
    cancel = threading.Event()
    original = sdk.GripperCtrl

    def cancel_after_gripper_enable(*values):
        original(*values)
        cancel.set()

    sdk.GripperCtrl = cancel_after_gripper_enable
    with pytest.raises(PiperHardwareError, match="cancelled"):
        hardware.connect(cancel_event=cancel)
    assert sdk.calls[-2:] == [("stop", 1, 0, 0), ("disconnect",)]
    assert not hardware.status()["connected"]


@pytest.mark.parametrize("failed_joint, fresh_raw", [(1, -265), (2, 1098), (1, 0), (2, 0), (None, None)])
def test_recorded_home_prefix_reports_constraint_conflict_without_sending_eighth_target(failed_joint, fresh_raw):
    # Exact successful setpoints and post-command feedback from home_run_01,
    # 2026-11-07. The failed eighth sample was NOT recorded; parameter values
    # below are distinct synthetic hypotheses, not reconstructed measurements.
    applied_rows = np.array([
        [2836, -1430, 2263, -1706, 21504, -2328, 0],
        [2836, -1264, 2097, -1706, 21504, -2328, 0],
        [2836, -1098, 1931, -1706, 21504, -2328, 0],
        [2836, -932, 1765, -1706, 21504, -2328, 0],
        [2836, -766, 1599, -1706, 21504, -2328, 0],
        [2808, -600, 1433, -1725, 21619, -2283, 0],
        [2781, -434, 1267, -1745, 21733, -2238, 0],
    ])
    observed_rows = np.array([
        [2863, -1596, 2429, -1687, 21390, -2373, 0],
        [2863, -1596, 2429, -1687, 21390, -2373, 0],
        [2863, -1596, 2429, -1687, 21390, -2373, 0],
        [2863, -1307, 2159, -1687, 21536, -2373, 0],
        [2863, -1027, 1996, -1687, 21753, -2373, 0],
        [2863, -737, 1768, -1687, 22029, -2373, 0],
        [2863, -481, 1379, -1687, 22029, -2373, 0],
    ])
    hardware, sdk, clock = startup_hardware(
        max_joint_velocity_rad_s=(np.deg2rad(5),) * 6,
        startup_recovery_feedback_slack_rad=np.deg2rad([.002, .25, .25, .002, .002, .002]))
    sdk.values = observed_rows[0].copy()
    hardware.connect()
    for sent, observed in zip(applied_rows, observed_rows):
        target = sent.astype(float)
        target[1:3] = 0
        clock.sleep(1 / 30)
        actual = hardware.command(target / hardware._sdk_scale)
        np.testing.assert_array_equal(np.rint(actual * hardware._sdk_scale), sent)
        sdk.values = observed.copy()
        hardware.read_state()
    if failed_joint is not None:
        sdk.values[failed_joint] = fresh_raw
    clock.sleep(1 / 30)
    if failed_joint is None:
        # With the last known feedback unchanged, the same eighth goal passes.
        # The recorded prefix alone therefore does not determine the failure.
        applied = hardware.command(target / hardware._sdk_scale)
        np.testing.assert_array_equal(np.rint(applied[1:3] * hardware._sdk_scale[1:3]), [-268, 1101])
        assert not hardware.status()["stopped"]
        return
    with pytest.raises(PiperHardwareError, match="projected setpoint") as failure:
        hardware.command(target / hardware._sdk_scale)
    diagnostics = failure.value.diagnostics
    assert diagnostics["offending_indices"] == [failed_joint]
    assert diagnostics["candidate_raw"][1:3] == [-268, 1101]
    assert diagnostics["baseline_raw"][1:3] == [-434, 1267]
    assert diagnostics["allowed_step_raw"][:6] == [166] * 6
    assert diagnostics["dt_s"] == pytest.approx(1 / 30, abs=1e-12)
    assert diagnostics["period_s"] == 1 / 30
    assert diagnostics["measured_si"][failed_joint] == fresh_raw / hardware._sdk_scale[failed_joint]
    if fresh_raw == 0:
        assert diagnostics["recovery_direction"][failed_joint] == 0
    elif failed_joint == 1:
        assert diagnostics["effective_lower_raw"][1] == -267
    else:
        assert diagnostics["effective_upper_raw"][2] == 1100
    assert "diagnostics=" in str(failure.value)
    assert hardware.status()["last_error"] == str(failure.value)
    assert len([call for call in sdk.calls if call[0] == "joints"]) == 7
    assert sdk.calls[-1] == ("stop", 1, 0, 0)


def native_hardware(**config):
    """A device planner advances feedback toward the last full MOVEJ endpoint."""
    config.setdefault("startup_recovery_feedback_slack_rad",
                      np.deg2rad([.002, .25, .25, .002, .002, .002]))
    hardware, sdk, clock = startup_hardware(**config)
    sdk.values[6] = 20000
    hardware.connect()
    def advance(duration):
        clock.sleep(duration)
        goals = [call[1:] for call in sdk.calls if call[0] == "joints"]
        if goals:
            step = max(1, round(1700 * duration))  # 1.7 degree/s simulated physical motion.
            sdk.values[:6] += np.clip(np.array(goals[-1]) - sdk.values[:6], -step, step)
    hardware._sleep = advance
    return hardware, sdk, clock, advance


def test_native_recovery_sends_legal_complete_goals_and_rebases_from_stable_feedback():
    import json
    hardware, sdk, clock, _ = native_hardware()
    sdk.values[4] += 20  # Native target must use fresh feedback, not connect's cached pose.
    fresh = sdk.values.copy()
    updates = []
    result = hardware.recover_startup_native(
        verified_max_joint_speed_rad_s=[3.] * 6, progress_callback=updates.append)
    assert result["completed"] is True
    assert result["stable_elapsed_s"] >= .5
    assert result["recovery_joint_indices"] == [1, 2]
    assert result["check_inward_progress"] is True
    assert all(update["check_inward_progress"] is True for update in updates)
    assert updates and all(update["measured_speed_rad_s"] is None
                           or len(update["measured_speed_rad_s"]) == 6 for update in updates)
    json.dumps(result, allow_nan=False)
    goal = fresh.copy()
    goal[1:3] = [1000, -1000]
    joint_calls = [call[1:] for call in sdk.calls if call[0] == "joints"]
    np.testing.assert_array_equal(joint_calls[0], goal[:6])
    for sent in joint_calls:
        assert np.all(np.array(sent) >= np.ceil(hardware._lower[:6] * hardware._sdk_scale[:6]))
        assert np.all(np.array(sent) <= np.floor(hardware._upper[:6] * hardware._sdk_scale[:6]))
    assert all(call == ("mode", 1, 1, 1, 0) for call in sdk.calls if call[0] == "mode")
    assert all(call[1] == 20000 for call in sdk.calls if call[0] == "gripper")
    assert not any(call[0] == "stop" for call in sdk.calls)
    assert hardware.status()["startup_recovery"]["active_joint_indices"] == []
    assert hardware.status()["startup_recovery"]["initial_state"] is None
    assert not hardware._native_recovery_retained.any()
    np.testing.assert_allclose(hardware._last_command, hardware.read_state(), atol=1e-12)
    clock.sleep(1 / hardware.config.control_hz)
    hardware.command(hardware.read_state())
    # Native recovery does not reopen nominal limits after arrival.
    sdk.values[1] = -1
    with pytest.raises(PiperFeedbackError, match="outside configured"):
        hardware.read_state()
    assert sdk.calls[-1] == ("stop", 1, 0, 0)


@pytest.mark.parametrize("bad", [
    {"speed_percent": 2}, {"speed_percent": True},
    {"verified_max_joint_speed_rad_s": [9.] * 6},
    {"verified_max_joint_speed_rad_s": [3.] * 5},
    {"verified_max_joint_speed_rad_s": [float("nan")] * 6},
    {"verified_max_joint_speed_rad_s": [0.] * 6},
    {"inset_rad": np.deg2rad(1.01)}, {"max_travel_rad": np.deg2rad(4.01)},
    {"measured_speed_limit_rad_s": np.deg2rad(5.01)},
    {"nonrecover_drift_rad": np.deg2rad(1.01)}, {"timeout_s": 20.01},
    {"stable_s": .49}, {"stable_s": 20.}, {"tolerance_rad": np.deg2rad(.251)},
    {"tolerance_rad": np.deg2rad(.25), "inset_rad": np.deg2rad(.2)},
    {"cancel_event": object()}, {"progress_callback": True},
])
def test_native_recovery_validates_all_options_before_control_send(bad):
    hardware, sdk, _, _ = native_hardware()
    before = list(sdk.calls)
    options = {"verified_max_joint_speed_rad_s": [3.] * 6, **bad}
    with pytest.raises(ValueError):
        hardware.recover_startup_native(**options)
    assert sdk.calls == before


@pytest.mark.parametrize("value", [None, 0, 1, "false", [], {}, np.bool_(False)])
def test_native_progress_check_requires_boolean_before_sdk_access(value):
    hardware, sdk, _ = startup_hardware()
    with pytest.raises(ValueError, match="check_inward_progress"):
        hardware.recover_startup_native(verified_max_joint_speed_rad_s=[3.] * 6,
                                        check_inward_progress=value)
    assert sdk.calls == []


def test_native_recovery_rejects_goal_travel_before_joint_or_mode_send():
    hardware, sdk, _, _ = native_hardware()
    with pytest.raises(PiperHardwareError, match="goal exceeds maximum travel"):
        hardware.recover_startup_native(verified_max_joint_speed_rad_s=[3.] * 6,
                                        max_travel_rad=np.deg2rad(3))
    assert not any(call[0] in ("joints", "mode") for call in sdk.calls)
    assert sdk.calls[-1] == ("stop", 1, 0, 0)


def test_native_recovery_rejects_an_initial_excursion_on_other_axis():
    hardware, sdk, clock = startup_hardware(
        joint_startup_recovery_tolerance_rad=np.deg2rad([3, 3, 3, 0, 0, 0]))
    sdk.values[0] = -155000
    hardware.connect()
    with pytest.raises(PiperHardwareError, match="only initial J2/J3"):
        hardware.recover_startup_native(verified_max_joint_speed_rad_s=[3.] * 6)
    assert not any(call[0] in ("joints", "mode") for call in sdk.calls)


def test_native_recovery_nominal_start_is_noop_and_requires_explicit_actuation():
    hardware, sdk, _ = make_hardware(actuation_enabled=True)
    hardware.connect()
    calls = list(sdk.calls)
    result = hardware.recover_startup_native(verified_max_joint_speed_rad_s=[3.] * 6)
    assert result["completed"] and result["recovery_joint_indices"] == []
    assert sdk.calls == calls
    with pytest.raises(PiperHardwareError, match="precede ordinary commands"):
        hardware.recover_startup_native(verified_max_joint_speed_rad_s=[3.] * 6)
    readonly, _, _ = make_hardware()
    readonly.connect()
    with pytest.raises(PiperHardwareError, match="actuation is disabled"):
        readonly.recover_startup_native(verified_max_joint_speed_rad_s=[3.] * 6)


@pytest.mark.parametrize("check_inward_progress,problem,match", [
    (enabled, problem, match)
    for enabled in (True, False)
    for problem, match in [
    ("speed", "measured speed exceeded"), ("travel", "exceeded total travel"),
    ("joint_drift", "nonrecovering joint drift"), ("grip_drift", "gripper drift"),
    ("reverse_nominal", "reversed inward progress"),
    ("nominal_reexit", "feedback moved outward"), ("fault", "arm_status"),
    ("stale", "stale"), ("timeout", "timed out"),
    ] if enabled or problem not in ("reverse_nominal", "nominal_reexit")
])
def test_native_recovery_stops_for_observed_motion_and_feedback_failures(
        check_inward_progress, problem, match):
    hardware, sdk, clock, advance = native_hardware()
    started = clock.now
    def disturbed(duration):
        advance(duration)
        elapsed = clock.now - started
        if problem == "speed":
            # Legal target, but >5 degree/s over a complete >=0.2 s feedback window.
            sdk.values[1] = min(1000, -1507 + round(8000 * elapsed))
            sdk.values[2] = max(-1000, 2381 - round(8000 * elapsed))
        elif problem == "travel":
            sdk.values[1] = 3000
        elif problem == "joint_drift":
            sdk.values[0] += 1001
        elif problem == "grip_drift":
            sdk.values[6] += 1001
        elif problem == "reverse_nominal":
            # Move slowly into nominal, then reverse >0.25 degree while still nominal.
            if sdk.values[1] >= 700:
                sdk.values[1] = 400
        elif problem == "nominal_reexit" and sdk.values[1] >= 0:
            sdk.values[1] = -251 if getattr(sdk, "entered", False) else 0
            sdk.entered = True
        elif problem == "fault":
            sdk.arm_status = 4
        elif problem == "stale":
            sdk.fixed_stamps["joints"] = sdk.fixed_stamps.get("joints", clock.wall())
        elif problem == "timeout":
            sdk.values[1:3] = [-1507, 2381]
    hardware._sleep = disturbed
    with pytest.raises(PiperHardwareError, match=match):
        hardware.recover_startup_native(verified_max_joint_speed_rad_s=[3.] * 6,
                                       check_inward_progress=check_inward_progress, timeout_s=1.
                                       if problem == "timeout" else 20.)
    assert sdk.calls[-1] == ("stop", 1, 0, 0)
    assert hardware.status()["stopped"]
    assert not hardware._native_recovery_retained.any()
    assert hardware.status()["startup_recovery"]["check_inward_progress"] is True


@pytest.mark.parametrize("check_inward_progress", [True, False])
def test_native_recovery_cancellation_between_mode_and_joints_sends_no_joint_target(check_inward_progress):
    hardware, sdk, _, _ = native_hardware()
    cancel = threading.Event()
    original = sdk.MotionCtrl_2
    def cancel_on_mode(*args):
        original(*args)
        cancel.set()
    sdk.MotionCtrl_2 = cancel_on_mode
    with pytest.raises(PiperHardwareError, match="cancelled"):
        hardware.recover_startup_native(verified_max_joint_speed_rad_s=[3.] * 6, cancel_event=cancel,
                                       check_inward_progress=check_inward_progress)
    assert not any(call[0] == "joints" for call in sdk.calls)
    assert sdk.calls[-1] == ("stop", 1, 0, 0)
    assert hardware.status()["startup_recovery"]["check_inward_progress"] is True


def test_native_recovery_progress_callback_failure_stops_without_resending():
    hardware, sdk, _, _ = native_hardware()
    def fail(snapshot):
        raise RuntimeError("logging failed")
    with pytest.raises(RuntimeError, match="logging failed"):
        hardware.recover_startup_native(verified_max_joint_speed_rad_s=[3.] * 6,
                                        progress_callback=fail)
    assert len([call for call in sdk.calls if call[0] == "joints"]) == 1
    assert sdk.calls[-1] == ("stop", 1, 0, 0)


def test_native_recovery_uses_pair_feedback_time_instead_of_cached_poll_intervals():
    hardware, sdk, clock, _ = native_hardware()
    baseline, baseline_wall = clock.now, clock.wall()
    original_get, original_emit = sdk.GetArmJointMsgs, sdk.emit
    cached = {"bucket": -1, "values": None}
    def sampled_emit(*ids):
        for can_id in ids:
            stamp = (sdk.fixed_stamps["joints"] if can_id in (0x2A5, 0x2A6, 0x2A7)
                     else clock.wall())
            sdk.bus.callback_function(NS(arbitration_id=can_id, timestamp=stamp, dlc=8, data=bytes(8)))
    def sampled_joints():
        bucket = int((clock.now - baseline + 1e-9) / .1)
        if bucket != cached["bucket"]:
            cached["bucket"], cached["values"] = bucket, sdk.values.copy()
        sdk.fixed_stamps["joints"] = baseline_wall + bucket * .1
        actual = sdk.values
        sdk.values = cached["values"]
        try:
            return original_get()
        finally:
            sdk.values = actual
    sdk.GetArmJointMsgs, sdk.emit = sampled_joints, sampled_emit
    updates = []
    result = hardware.recover_startup_native(verified_max_joint_speed_rad_s=[3.] * 6,
                                             progress_callback=updates.append)
    assert result["completed"]
    intervals = np.diff([update["elapsed_s"] for update in updates])
    assert np.all(intervals > .079)
    speeds = [update["measured_speed_rad_s"] for update in updates
              if update["measured_speed_rad_s"] is not None]
    assert speeds and np.max(speeds) < np.deg2rad(2)


def test_native_final_hold_refreshes_pose_after_progress_callback():
    hardware, sdk, _, _ = native_hardware()
    initial_j1 = sdk.values[0]
    changed = []
    def change_during_callback(snapshot):
        if snapshot["stable_elapsed_s"] >= .5 and not changed:
            sdk.values[0] += 100  # 0.1 degree, still within every arrival/drift guard.
            changed.append(True)
    result = hardware.recover_startup_native(verified_max_joint_speed_rad_s=[3.] * 6,
                                             progress_callback=change_during_callback)
    assert result["completed"] and changed
    assert [call for call in sdk.calls if call[0] == "joints"][-1][1] == initial_j1 + 100
    hardware.read_state()
    hardware.command(hardware.read_state())
    hardware.hold()
    hardware.close()
    assert sdk.calls[-1] == ("disconnect",)
    assert not any(call[0] == "stop" for call in sdk.calls)


def test_native_final_hold_rechecks_motion_guard_after_callback():
    hardware, sdk, _, _ = native_hardware()
    def change_during_callback(snapshot):
        if snapshot["stable_elapsed_s"] >= .5:
            sdk.values[6] += 1001
    with pytest.raises(PiperHardwareError, match="gripper drift"):
        hardware.recover_startup_native(verified_max_joint_speed_rad_s=[3.] * 6,
                                        progress_callback=change_during_callback)
    assert sdk.calls[-1] == ("stop", 1, 0, 0)
    assert all(call[1] == 20000 for call in sdk.calls if call[0] == "gripper")


def test_native_slow_callback_cannot_send_after_deadline():
    hardware, sdk, clock, _ = native_hardware()
    def slow_callback(snapshot):
        clock.sleep(21.)
    with pytest.raises(PiperHardwareError, match="timed out"):
        hardware.recover_startup_native(verified_max_joint_speed_rad_s=[3.] * 6,
                                        progress_callback=slow_callback)
    assert len([call for call in sdk.calls if call[0] == "joints"]) == 1
    assert sdk.calls[-1] == ("stop", 1, 0, 0)


def test_native_external_stop_latch_prevents_resending_after_callback():
    hardware, sdk, _, _ = native_hardware()
    with pytest.raises(PiperHardwareError, match="stop is latched"):
        hardware.recover_startup_native(verified_max_joint_speed_rad_s=[3.] * 6,
                                        progress_callback=lambda update: hardware.stop())
    assert len([call for call in sdk.calls if call[0] == "joints"]) == 1
    assert sdk.calls[-1] == ("stop", 1, 0, 0)


# Recorded native prefix: home_native_run_01/replay.jsonl, 2026-11-07.
# Columns: recorded native elapsed seconds, raw J2/J3/J5 (millidegrees).
# Other joints/gripper stayed [2861, _, _, 0, _, -2346, 0].
NATIVE_BOUNDARY_CAPTURE = [
    (0.020969680, -1508, 1959, 23339),
    (0.042124005, -1508, 1959, 23339),
    (0.063762461, -1508, 1959, 23339),
    (0.085243852, -1508, 1959, 23339),
    (0.106348716, -1508, 1959, 23339),
    (0.127469097, -1402, 1849, 23339),
    (0.149421541, -1402, 1849, 23339),
    (0.170486111, -1402, 1849, 23339),
    (0.191461357, -1294, 1747, 23339),
    (0.212714696, -1294, 1755, 23444),
    (0.234593884, -1294, 1755, 23444),
    (0.255937229, -1187, 1755, 23444),
    (0.277177550, -1186, 1636, 23444),
    (0.298605027, -1186, 1637, 23547),
    (0.320300473, -1064, 1637, 23549),
    (0.341516833, -1061, 1637, 23549),
    (0.362773979, -1061, 1506, 23549),
    (0.384831959, -1061, 1506, 23644),
    (0.405924503, -950, 1506, 23644),
    (0.427074389, -950, 1391, 23644),
    (0.448617458, -863, 1389, 23644),
    (0.470336759, -863, 1389, 23644),
    (0.491400751, -756, 1274, 23644),
    (0.512521503, -724, 1274, 23644),
    (0.534269751, -730, 1274, 23644),
    (0.555716825, -705, 1159, 23644),
    (0.576802715, -605, 1159, 23514),
    (0.597881060, -608, 1042, 23514),
    (0.619374057, -608, 1062, 23514),
    (0.640598186, -500, 1062, 23514),
    (0.661764907, -509, 1025, 23514),
    (0.682877362, -491, 925, 23411),
    (0.704276476, -376, 824, 23427),
    (0.725537693, -376, 790, 23427),
    (0.746902106, -376, 795, 23427),
    (0.767992483, -265, 795, 23427),
    (0.789597936, -265, 690, 23427),
    (0.810666351, -265, 698, 23427),
    (0.831829461, -160, 698, 23427),
    (0.853036587, -160, 579, 23427),
    (0.874516841, -56, 579, 23427),
    (0.895574659, 0, 476, 23427),
    (0.916638799, 0, 485, 23427),
]


def test_native_recorded_boundary_recross_is_accepted_without_reopening_after_failure():
    """Real 43-sample prefix + captured -52 raw J2; no invented success tail."""
    hardware, sdk, clock = startup_hardware(
        startup_recovery_feedback_slack_rad=np.deg2rad([.002, .25, .25, .002, .002, .002]))
    sdk.values = np.array([2861, -1508, 1959, 0, 23339, -2346, 0])
    hardware.connect()
    started = clock.now
    # The next raw 0x2a5 at 1794041962.9413748 contains J2=-52,
    # 0.001550 s after the last JSONL row, before software stop status.
    sequence = iter([*NATIVE_BOUNDARY_CAPTURE, (.9181888, -52, 485, 23427)])
    updates = []
    def next_record(duration):
        elapsed, j2, j3, j5 = next(sequence)
        clock.now = started + elapsed
        sdk.values[[1, 2, 4]] = [j2, j3, j5]
    def record_and_cancel(update):
        updates.append(update)
        if len(updates) == 44:
            # Ends this recorded trace deliberately; it does not assert that
            # real hardware would subsequently arrive at the native goal.
            raise RuntimeError("recorded prefix accepted")
    hardware._sleep = next_record
    with pytest.raises(RuntimeError, match="recorded prefix accepted"):
        hardware.recover_startup_native(verified_max_joint_speed_rad_s=[3.] * 6,
                                        progress_callback=record_and_cancel)
    assert len(updates) == 44
    assert np.rint(updates[-1]["state"][1] * hardware._sdk_scale[1]) == -52
    assert not hardware._native_recovery_retained.any()
    assert hardware._recovery_direction[1] == 0  # Native scope exit retires nominal-seen J2.
    for call in sdk.calls:
        if call[0] == "joints":
            assert call[2:4] == (1000, -1000)  # Never transmit outside targets.
    assert sdk.calls[-1] == ("stop", 1, 0, 0)
    with pytest.raises(PiperFeedbackError, match="outside configured"):
        hardware.read_state()


def test_ordinary_recovery_still_retires_at_first_zero_for_same_recorded_prefix():
    hardware, sdk, clock = startup_hardware(
        startup_recovery_feedback_slack_rad=np.deg2rad([.002, .25, .25, .002, .002, .002]))
    sdk.values = np.array([2861, -1508, 1959, 0, 23339, -2346, 0])
    hardware.connect()
    started = clock.now
    for elapsed, j2, j3, j5 in NATIVE_BOUNDARY_CAPTURE:
        clock.now = started + elapsed
        sdk.values[[1, 2, 4]] = [j2, j3, j5]
        hardware.read_state()
    sdk.values[1] = -52
    with pytest.raises(PiperFeedbackError, match="outside configured") as failure:
        hardware.read_state()
    assert failure.value.diagnostics["offending_indices"] == [1]
    assert failure.value.diagnostics["native_retained_indices"] == []
    assert not any(call[0] in ("mode", "joints") for call in sdk.calls)


def test_native_boundary_allowance_keeps_exact_slack_and_reports_failed_sample():
    hardware, sdk, clock, _ = native_hardware()
    sequence = iter([0, -250, -251])
    def boundary_samples(duration):
        clock.sleep(duration)
        sdk.values[1] = next(sequence)
    hardware._sleep = boundary_samples
    with pytest.raises(PiperFeedbackError, match="feedback moved outward") as failure:
        hardware.recover_startup_native(verified_max_joint_speed_rad_s=[3.] * 6)
    diagnostics = failure.value.diagnostics
    assert diagnostics["offending_indices"] == [1]
    assert diagnostics["native_retained_indices"] == [1, 2]
    assert diagnostics["best_excursion_rad"][1] == 0
    assert diagnostics["allowed_excursion_rad"][1] == pytest.approx(np.deg2rad(.25))
    assert diagnostics["measured_si"][1] == pytest.approx(np.deg2rad(-.251))
    assert "diagnostics=" in str(failure.value)
    assert not hardware._native_recovery_retained.any()
    assert hardware._recovery_direction[1] == 0
    assert sdk.calls[-1] == ("stop", 1, 0, 0)


def test_native_retention_starts_before_its_first_fresh_nominal_sample():
    hardware, sdk, _, _ = native_hardware()
    sdk.values[1] = 0  # Becomes nominal on native's first read, after connect.
    result = hardware.recover_startup_native(verified_max_joint_speed_rad_s=[3.] * 6)
    assert result["completed"] and result["recovery_joint_indices"] == [1, 2]
    assert result["target"][1] == pytest.approx(np.deg2rad(1))
    assert not hardware._native_recovery_retained.any()


@pytest.mark.parametrize("check_inward_progress", [True, False])
def test_native_does_not_reopen_a_joint_retired_before_method_entry(check_inward_progress):
    hardware, sdk, _, _ = native_hardware()
    sdk.values[1] = 0
    hardware.read_state()  # Ordinary read retires J2 before native starts.
    sdk.values[1] = -52
    with pytest.raises(PiperFeedbackError, match="outside configured") as failure:
        hardware.recover_startup_native(verified_max_joint_speed_rad_s=[3.] * 6,
                                       check_inward_progress=check_inward_progress)
    assert failure.value.diagnostics["native_retained_indices"] == [2]
    assert hardware._recovery_direction[1] == 0
    assert not hardware._native_recovery_retained.any()
    assert not any(call[0] in ("mode", "joints") for call in sdk.calls)


@pytest.mark.parametrize("check_inward_progress", [True, False])
def test_native_retention_does_not_allow_a_new_nonrecovering_joint_excursion(check_inward_progress):
    hardware, sdk, clock, _ = native_hardware()
    sdk.values[0] = round(hardware._lower[0] * hardware._sdk_scale[0])
    def unrelated_excursion(duration):
        clock.sleep(duration)
        sdk.values[0] -= 1
    hardware._sleep = unrelated_excursion
    with pytest.raises(PiperFeedbackError, match="outside configured") as failure:
        hardware.recover_startup_native(verified_max_joint_speed_rad_s=[3.] * 6,
                                       check_inward_progress=check_inward_progress)
    assert failure.value.diagnostics["offending_indices"] == [0]
    assert failure.value.diagnostics["native_retained_indices"] == [1, 2]
    assert not hardware._native_recovery_retained.any()


@pytest.mark.parametrize("check_inward_progress", [True, False])
def test_native_first_fresh_sample_beyond_inset_cannot_request_an_outward_goal(check_inward_progress):
    hardware, sdk, _, _ = native_hardware()
    sdk.values[1] = 1500  # Still active before native's fresh read; already past its +1 degree goal.
    with pytest.raises(PiperHardwareError, match="goal would reverse inward progress"):
        hardware.recover_startup_native(verified_max_joint_speed_rad_s=[3.] * 6,
                                       check_inward_progress=check_inward_progress)
    assert not any(call[0] in ("mode", "joints") for call in sdk.calls)
    assert not hardware._native_recovery_retained.any()
    assert hardware._recovery_direction[1] == 0
    assert sdk.calls[-1] == ("stop", 1, 0, 0)


@pytest.mark.parametrize("check_inward_progress", [True, False])
def test_native_recorded_j2_jump_only_exempts_position_progress_check(check_inward_progress):
    """The two recorded raw positions are not a successful physical recovery trace."""
    hardware, sdk, clock = startup_hardware(
        startup_recovery_feedback_slack_rad=np.deg2rad([.002, .25, .25, .002, .002, .002]))
    sdk.values = np.array([4633, -633, 2336, -353, 20952, 2725, 27400])
    hardware.connect()
    updates = []

    def recorded_jump(duration):
        clock.sleep(duration)
        sdk.values[[1, 4]] = [-1676, 21499]

    def stop_after_recorded_prefix(update):
        updates.append(update)
        assert hardware.status()["startup_recovery"]["check_inward_progress"] is False
        raise RuntimeError("recorded prefix accepted; no completion asserted")

    hardware._sleep = recorded_jump
    error = PiperFeedbackError if check_inward_progress else RuntimeError
    match = "feedback moved outward" if check_inward_progress else "recorded prefix accepted"
    with pytest.raises(error, match=match):
        hardware.recover_startup_native(
            verified_max_joint_speed_rad_s=[3.] * 6,
            check_inward_progress=check_inward_progress,
            progress_callback=stop_after_recorded_prefix)
    assert len(updates) == (0 if check_inward_progress else 1)
    if updates:
        assert updates[0]["state"][1] == pytest.approx(np.deg2rad(-1.676))
        assert updates[0]["measured_speed_rad_s"] is None  # Less than 0.2 s of feedback.
        assert updates[0]["check_inward_progress"] is False
        assert updates[0]["completed"] is False
    assert hardware.status()["startup_recovery"]["check_inward_progress"] is True
    assert not hardware._native_recovery_retained.any()
    assert sdk.calls[-1] == ("stop", 1, 0, 0)
    # Outside the method, ordinary feedback once again uses the shrinking bound.
    with pytest.raises(PiperFeedbackError, match="feedback moved outward"):
        hardware.read_state()


@pytest.mark.parametrize("check_inward_progress", [True, False])
def test_native_slow_reversal_uses_optional_progress_policy(check_inward_progress):
    hardware, sdk, clock, _ = native_hardware()
    started = clock.now
    updates = []

    def slow_reversal(duration):
        clock.sleep(duration)
        elapsed = clock.now - started
        # A synthetic motion at < 2 degree/s reverses 0.6 degrees before arriving.
        sdk.values[1] = round(np.interp(elapsed, [0, .5, 1, 1.5, 2, 3],
                                       [-1507, -700, -1300, -700, 0, 1000]))
        sdk.values[2] = round(max(-1000, 2381 - 1700 * elapsed))

    def record(update):
        updates.append(update)
        assert hardware.status()["startup_recovery"]["check_inward_progress"] is check_inward_progress

    hardware._sleep = slow_reversal
    kwargs = dict(verified_max_joint_speed_rad_s=[3.] * 6,
                  check_inward_progress=check_inward_progress, progress_callback=record)
    if check_inward_progress:
        with pytest.raises(PiperFeedbackError, match="feedback moved outward"):
            hardware.recover_startup_native(**kwargs)
        assert sdk.calls[-1] == ("stop", 1, 0, 0)
    else:
        result = hardware.recover_startup_native(**kwargs)
        assert result["completed"] and result["stable_elapsed_s"] >= .5
        assert result["check_inward_progress"] is False
        assert all(update["check_inward_progress"] is False for update in updates)
        assert not any(call[0] == "stop" for call in sdk.calls)
        # Completing native recovery permanently retires the former exceptions.
        sdk.values[1] = -1
        with pytest.raises(PiperFeedbackError, match="outside configured"):
            hardware.read_state()
    assert hardware.status()["startup_recovery"]["check_inward_progress"] is True
    assert not hardware._native_recovery_retained.any()


@pytest.mark.parametrize("tolerance_deg", [2.5, 3])
def test_native_without_progress_check_retains_configured_feedback_cap(tolerance_deg):
    hardware, sdk, clock, _ = native_hardware(
        joint_startup_recovery_tolerance_rad=np.deg2rad([0, tolerance_deg, 3, 0, 0, 0]))

    def exceed_cap(duration):
        clock.sleep(duration)
        sdk.values[1] = -round(tolerance_deg * 1000) - 1

    hardware._sleep = exceed_cap
    with pytest.raises(PiperFeedbackError, match="configured startup tolerance") as failure:
        hardware.recover_startup_native(verified_max_joint_speed_rad_s=[3.] * 6,
                                       check_inward_progress=False)
    assert failure.value.diagnostics["offending_indices"] == [1]
    assert failure.value.diagnostics["allowed_excursion_rad"][1] == pytest.approx(np.deg2rad(tolerance_deg))
    assert hardware.status()["startup_recovery"]["check_inward_progress"] is True
    assert not hardware._native_recovery_retained.any()
    assert sdk.calls[-1] == ("stop", 1, 0, 0)


def test_gripper_feedback_decodes_one_signed_raw_frame_and_returns_defensive_snapshot():
    hardware, sdk, clock = make_hardware()
    hardware.connect()
    sdk.gripper_enabled = True
    sdk.gripper_status_extra = 0x80  # Homed is not a fault.
    sdk.gripper_effort = -1234
    before = list(sdk.calls)
    snapshot = hardware.read_gripper_feedback()
    assert snapshot == dict(width_m=.04, effort_raw=-1234, status_raw=0xC0,
                            enabled=True, fault=False, timestamp_s=clock.wall())
    assert isinstance(snapshot["effort_raw"], int) and isinstance(snapshot["status_raw"], int)
    assert sdk.calls == before
    snapshot["effort_raw"], snapshot["width_m"] = 999, 123
    assert hardware.read_gripper_feedback()["effort_raw"] == -1234
    sdk.gripper_effort, sdk.values[6] = 4321, 30000
    next_snapshot = hardware.read_gripper_feedback()
    assert next_snapshot["effort_raw"] == 4321 and next_snapshot["width_m"] == .03
    assert snapshot["effort_raw"] == 999


def test_gripper_feedback_does_not_mix_mutable_sdk_fields_with_raw_frame():
    hardware, sdk, clock = make_hardware()
    hardware.connect()
    sdk.gripper_enabled, sdk.gripper_effort, sdk.gripper_status_extra = True, -1234, 0x80
    original_get = sdk.GetArmGripperMsgs
    def sdk_cache_changes_after_raw_receive():
        message = original_get()  # Immutable receive snapshot: width40000, effort-1234, status0xC0.
        message.gripper_state.grippers_angle = 12345
        message.gripper_state.grippers_effort = 999
        message.gripper_state.status_code = 1
        message.time_stamp = clock.wall() + .001  # A newer SDK sample need not equal raw snapshot.
        return message
    sdk.GetArmGripperMsgs = sdk_cache_changes_after_raw_receive
    result = hardware.read_gripper_feedback()
    assert result == dict(width_m=.04, effort_raw=-1234, status_raw=0xC0,
                          enabled=True, fault=False, timestamp_s=clock.wall())


def frozen_gripper_feedback():
    hardware, sdk, clock = make_hardware()
    hardware.connect()
    message = sdk.GetArmGripperMsgs()
    sdk.GetArmGripperMsgs = lambda: message
    return hardware, sdk, clock, message


def test_gripper_raw_payload_is_copied_before_caller_can_mutate_its_bytearray():
    hardware, sdk, clock, _ = frozen_gripper_feedback()
    payload = bytearray.fromhex("00009c40fb2ec000")
    sdk.bus.callback_function(NS(arbitration_id=0x2A8, timestamp=clock.wall(), dlc=8, data=payload))
    payload[:] = bytes(8)
    result = hardware.read_gripper_feedback()
    assert result["width_m"] == .04 and result["effort_raw"] == -1234
    assert result["enabled"] and result["status_raw"] == 0xC0


@pytest.mark.parametrize("bit", range(6))
def test_gripper_feedback_returns_each_fault_bit_without_any_control_side_effect(bit):
    hardware, sdk, _ = make_hardware(actuation_enabled=True)
    hardware.connect()
    sdk.gripper_status_extra = 1 << bit
    before = list(sdk.calls)
    result = hardware.read_gripper_feedback()
    assert result["fault"] is True and result["enabled"] is True
    assert result["status_raw"] == 0x40 | (1 << bit)
    assert sdk.calls == before
    assert hardware.status()["stopped"] is False


def test_gripper_feedback_reports_disabled_and_does_not_enable_it():
    hardware, sdk, _ = make_hardware()
    hardware.connect()
    before = list(sdk.calls)
    result = hardware.read_gripper_feedback()
    assert result["enabled"] is False and result["fault"] is False
    assert sdk.calls == before


@pytest.mark.parametrize("which", ["sdk", "raw", "receipt"])
def test_gripper_feedback_rejects_independent_staleness_without_sending_stop(which):
    hardware, sdk, clock, message = frozen_gripper_feedback()
    before = list(sdk.calls)
    if which == "sdk":
        message.time_stamp = clock.wall() - 1
    elif which == "raw":
        clock.sleep(.3)
        message.time_stamp = clock.wall()
    else:
        stamp, received, payload = hardware._gripper_frame
        hardware._gripper_frame = (stamp, received - 1, payload)
    with pytest.raises(PiperFeedbackError, match="[Ss]tale"):
        hardware.read_gripper_feedback()
    assert sdk.calls == before
    assert not hardware.status()["stopped"]


@pytest.mark.parametrize("malformed", ["missing", "short", "nan_timestamp", "remote", "error"])
def test_gripper_feedback_never_fabricates_missing_or_invalid_raw_frames(malformed):
    hardware, sdk, clock, _ = frozen_gripper_feedback()
    hardware._gripper_frame = None
    if malformed != "missing":
        message = NS(arbitration_id=0x2A8, timestamp=clock.wall(), dlc=8, data=bytes(8))
        if malformed == "short":
            message.dlc, message.data = 7, bytes(7)
        elif malformed == "nan_timestamp":
            message.timestamp = float("nan")
        elif malformed == "remote":
            message.is_remote_frame = True
        else:
            message.is_error_frame = True
        sdk.bus.callback_function(message)
    before = list(sdk.calls)
    with pytest.raises(PiperFeedbackError, match="Missing.*0x2a8"):
        hardware.read_gripper_feedback()
    assert sdk.calls == before


@pytest.mark.parametrize("field,value", [
    ("grippers_effort", float("nan")), ("grippers_effort", float("inf")),
    ("grippers_effort", None), ("grippers_effort", True), ("grippers_effort", 32768),
    ("grippers_effort", -32769), ("grippers_angle", 2**31), ("status_code", 256),
    ("status_code", -1), ("status_code", "64"),
])
def test_gripper_feedback_rejects_invalid_sdk_fields_without_fallback(field, value):
    hardware, sdk, _, message = frozen_gripper_feedback()
    setattr(message.gripper_state, field, value)
    before = list(sdk.calls)
    with pytest.raises(PiperFeedbackError, match="invalid gripper"):
        hardware.read_gripper_feedback()
    assert sdk.calls == before


def test_gripper_feedback_rejects_unsupported_sdk_effort_field_instead_of_default_zero():
    hardware, sdk, _, message = frozen_gripper_feedback()
    del message.gripper_state.grippers_effort
    before = list(sdk.calls)
    with pytest.raises(PiperFeedbackError, match="does not expose complete"):
        hardware.read_gripper_feedback()
    assert sdk.calls == before


@pytest.mark.parametrize("width", [-1, 70001])
def test_gripper_feedback_rejects_invalid_raw_width_without_touching_control(width):
    hardware, sdk, clock, _ = frozen_gripper_feedback()
    payload = width.to_bytes(4, "big", signed=True) + bytes(4)
    sdk.bus.callback_function(NS(arbitration_id=0x2A8, timestamp=clock.wall(), dlc=8, data=payload))
    before = list(sdk.calls)
    with pytest.raises(PiperFeedbackError, match="outside configured width"):
        hardware.read_gripper_feedback()
    assert sdk.calls == before


def test_gripper_feedback_preserves_duplicate_timestamps_and_rejects_rollback():
    hardware, sdk, clock, _ = frozen_gripper_feedback()
    first = hardware.read_gripper_feedback()
    assert hardware.read_gripper_feedback()["timestamp_s"] == first["timestamp_s"]
    payload = bytes.fromhex("00009c4000010000")
    sdk.bus.callback_function(NS(arbitration_id=0x2A8, timestamp=clock.wall() - .01,
                                dlc=8, data=payload))
    with pytest.raises(PiperFeedbackError, match="Invalid or stale"):
        hardware.read_gripper_feedback()


def test_gripper_feedback_requires_connection_and_clears_snapshot_when_disconnected():
    hardware, sdk, _ = make_hardware()
    with pytest.raises(PiperHardwareError, match="not connected"):
        hardware.read_gripper_feedback()
    assert sdk.calls == []
    hardware.connect()
    hardware.read_gripper_feedback()
    assert hardware._gripper_frame is not None
    hardware.close()
    assert hardware._gripper_frame is None
    with pytest.raises(PiperHardwareError, match="not connected"):
        hardware.read_gripper_feedback()


def applied_grasp_hardware():
    hardware, sdk, clock = make_hardware(actuation_enabled=True, max_gripper_velocity_m_s=.6)
    hardware.connect()
    target = hardware.read_state()
    target[6] = .03
    applied = hardware.command(target)
    assert applied[6] == .03
    sdk.values[6] = 35000  # A contacted object keeps measured opening above the applied goal.
    return hardware, sdk, clock, applied


def test_hold_grasp_uses_fresh_arm_and_keeps_exact_applied_clamp_through_close():
    hardware, sdk, _, previous = applied_grasp_hardware()
    sdk.values[0] = 11000
    before = len(sdk.calls)
    result = hardware.hold_grasp(gripper_target_m=previous[6] - .9e-6)
    assert result[0] == pytest.approx(np.deg2rad(11))
    assert result[6] == .03  # The confirmation tolerance cannot request extra closing.
    sdk.values[0], sdk.values[6] = 12000, 36000
    hardware.close()
    subsequent = sdk.calls[before:]
    assert [call[1] for call in subsequent if call[0] == "gripper"] == [30000, 30000]
    assert [call for call in subsequent if call[0] == "joints"][-1][1] == 12000
    assert all(call[2:] == (1000, 1, 0) for call in subsequent if call[0] == "gripper")
    assert not any(call[0] in ("enable", "stop") for call in subsequent)
    assert sdk.calls[-1] == ("disconnect",)
    assert hardware._grasp_hold_target_m is None


@pytest.mark.parametrize("ordinary", ["command", "hold"])
def test_ordinary_action_clears_grasp_preservation_and_restores_default_close(ordinary):
    hardware, sdk, clock, applied = applied_grasp_hardware()
    hardware.hold_grasp(gripper_target_m=applied[6])
    clock.sleep(1 / hardware.config.control_hz)
    if ordinary == "command":
        hardware.command(hardware.read_state())
    else:
        hardware.hold()
    assert hardware._grasp_hold_target_m is None
    sdk.values[6] = 38000
    hardware.close()
    assert [call for call in sdk.calls if call[0] == "gripper"][-1][1] == 38000


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, ".03", -.001, .071, .03000101])
def test_hold_grasp_rejects_invalid_or_different_goal_before_sending(value):
    hardware, sdk, _, _ = applied_grasp_hardware()
    before = list(sdk.calls)
    with pytest.raises(ValueError):
        hardware.hold_grasp(gripper_target_m=value)
    assert sdk.calls == before
    assert hardware._grasp_hold_target_m is None


def test_hold_grasp_requires_actuation_and_an_existing_applied_goal():
    hardware, sdk, _ = make_hardware(actuation_enabled=True)
    hardware.connect()
    before = list(sdk.calls)
    with pytest.raises(PiperHardwareError, match="previously applied"):
        hardware.hold_grasp(gripper_target_m=.04)
    assert sdk.calls == before
    readonly, readonly_sdk, _ = make_hardware()
    readonly.connect()
    before = list(readonly_sdk.calls)
    with pytest.raises(PiperHardwareError, match="actuation is disabled"):
        readonly.hold_grasp(gripper_target_m=.04)
    assert readonly_sdk.calls == before


@pytest.mark.parametrize("problem", ["stale", "fault", "stop"])
def test_grasp_hold_cleanup_never_resumes_after_fault_or_stop(problem):
    hardware, sdk, clock, applied = applied_grasp_hardware()
    hardware.hold_grasp(gripper_target_m=applied[6])
    before = len(sdk.calls)
    if problem == "stale":
        sdk.fixed_stamps["gripper"] = clock.wall() - 1
        with pytest.raises(PiperFeedbackError, match="stale"):
            hardware.hold_grasp(gripper_target_m=applied[6])
    elif problem == "fault":
        sdk.gripper_fault = True
        with pytest.raises(PiperFeedbackError, match="gripper driver reports a fault"):
            hardware.hold_grasp(gripper_target_m=applied[6])
    else:
        hardware.stop()
        with pytest.raises(PiperHardwareError, match="stop is latched"):
            hardware.hold_grasp(gripper_target_m=applied[6])
    assert hardware._grasp_hold_target_m is None
    hardware.close()
    assert sdk.calls[before:] == [("stop", 1, 0, 0), ("disconnect",)]


def test_grasp_close_still_checks_fresh_feedback_before_resending_clamp():
    hardware, sdk, clock, applied = applied_grasp_hardware()
    hardware.hold_grasp(gripper_target_m=applied[6])
    before = len(sdk.calls)
    sdk.fixed_stamps["joints"] = clock.wall() - 1
    with pytest.raises(PiperHardwareError, match="close failed to hold"):
        hardware.close()
    assert sdk.calls[before:] == [("stop", 1, 0, 0), ("disconnect",)]
