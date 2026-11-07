"""Configuration boundary tests for the opt-in reference PiPER adapter."""
from copy import deepcopy

import numpy as np
import pytest

from pressb.control.real_replay import ReplayPlan
from pressb.robots.piper.real import PiperHardwareConfig


def plan_data():
    return {
        "schema_version": 1, "source": {"kind": "test_reference_adapter"}, "fps": 30,
        "actions": [[0, .2, -.2, 0, .2, 0, .03]],
        "expected_initial_state": [0, .2, -.2, 0, np.deg2rad(76.276), 0, .03],
        "expected_initial_tolerance": [np.deg2rad(.5)] * 6 + [.0005],
        "robot": {"actuation_enabled": True, "control_hz": 30, "speed_percent": 1,
                  "joint_upper_rad": np.deg2rad([154, 195, 0, 102, 70, 170]).tolist()},
        "replay": {"execution_mode": "reference"},
    }


@pytest.mark.parametrize("policy", [None, "", "native", True, 1])
def test_invalid_policy_rejected_before_construction(policy):
    with pytest.raises(ValueError, match="command_policy"):
        PiperHardwareConfig(command_policy=policy)


def test_library_hardware_default_is_projected():
    assert PiperHardwareConfig().command_policy == "projected"


def test_reference_accepts_finite_outside_joint_pose_without_widening_targets():
    data = plan_data()
    original_robot = deepcopy(data["robot"])
    plan = ReplayPlan.from_mapping(data)
    assert plan.robot_config["command_policy"] == "reference"
    assert plan.robot_config["speed_percent"] == 1
    assert plan.expected_initial_state[4] == pytest.approx(np.deg2rad(76.276))
    assert plan.command_bounds[1][4] == pytest.approx(np.deg2rad(70))
    assert plan.robot_config["joint_upper_rad"] == original_robot["joint_upper_rad"]
    assert "command_policy" not in data["robot"]
    config = PiperHardwareConfig(**plan.robot_config)
    assert config.command_policy == "reference"
    dispatch = plan.summary()["reference_dispatch"]
    assert not dispatch["measured_joint_bounds_enforced"]
    assert "no adapter velocity slew guarantee" in dispatch["interpolation"]
    assert "no new target" in dispatch["cleanup"]


def test_tracked_still_rejects_same_outside_measured_pose():
    data = plan_data()
    data["replay"]["execution_mode"] = "tracked"
    with pytest.raises(ValueError, match="initial state exceeds"):
        ReplayPlan.from_mapping(data)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_reference_still_rejects_nonfinite_joint_feedback(value):
    data = plan_data()
    data["expected_initial_state"][4] = value
    with pytest.raises(ValueError, match="finite"):
        ReplayPlan.from_mapping(data)


@pytest.mark.parametrize("value", [-.001, .071])
def test_reference_still_checks_expected_gripper_range(value):
    data = plan_data()
    data["expected_initial_state"][6] = value
    with pytest.raises(ValueError, match="gripper state exceeds"):
        ReplayPlan.from_mapping(data)


@pytest.mark.parametrize("mode,policy", [("reference", "projected"), ("tracked", "reference")])
def test_explicit_inconsistent_adapter_policy_rejected(mode, policy):
    data = plan_data()
    data["replay"]["execution_mode"] = mode
    data["robot"]["command_policy"] = policy
    data["expected_initial_state"][4] = .2
    with pytest.raises(ValueError, match="replay requires robot.command_policy"):
        ReplayPlan.from_mapping(data)


def test_reference_policy_and_nominal_limits_survive_revalidation():
    plan = ReplayPlan.from_mapping(plan_data())
    copy = plan_data()
    copy["robot"] = plan.robot_config
    checked = ReplayPlan.from_mapping(copy)
    assert checked.robot_config == plan.robot_config
    assert checked.summary()["reference_dispatch"]["target_upper_si"][4] == pytest.approx(np.deg2rad(70))


def test_direct_reference_adapter_cannot_widen_original_target_ranges():
    from pressb.robots.piper.real import PiperHardware
    hardware = PiperHardware({
        "command_policy": "reference",
        "joint_lower_rad": [-4.] * 6, "joint_upper_rad": [4.] * 6,
        "gripper_max_width_m": .09,
    })
    description = hardware.describe()
    assert description["action"]["upper"][4] == pytest.approx(np.deg2rad(75))
    assert description["action"]["upper"][6] == .07
    assert "no adapter velocity slew guarantee" in description["interpolation"]
    value = np.array([0, .1, -.1, 0, np.deg2rad(76), 0, .03])
    with pytest.raises(ValueError, match="outside configured"):
        hardware.validate_target(value)


def test_reference_adapter_rejects_empty_nominal_intersection_before_can():
    from pressb.robots.piper.real import PiperHardware
    with pytest.raises(ValueError, match="must overlap"):
        PiperHardware({"command_policy": "reference",
                       "joint_lower_rad": [3.] * 6, "joint_upper_rad": [4.] * 6})


def test_reference_policy_does_not_silently_enter_native_recovery():
    from pressb.robots.piper.real import PiperHardware, PiperHardwareError
    hardware = PiperHardware({"command_policy": "reference"})
    with pytest.raises(PiperHardwareError, match="requires projected"):
        hardware.recover_startup_native(verified_max_joint_speed_rad_s=[1] * 6)
