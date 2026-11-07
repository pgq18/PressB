import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from pressb.control.real_replay import ReplayError, ReplayPlan, ReplayWatchdog, execute_replay


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, value):
        assert value >= 0
        self.now += value


@pytest.fixture
def plan():
    return {"schema_version": 1, "source": {"dataset": "fixture", "episode_index": 144},
            "fps": 30, "actions": [[0, .1, -.1, 0, .1, 0, .02], [0.01, .1, -.1, 0, .1, 0, .02]],
            "expected_initial_state": [0, .1, -.1, 0, .1, 0, .02],
            "expected_initial_tolerance": [math.radians(1)] * 6 + [.001],
            "robot": {"actuation_enabled": True, "control_hz": 30},
            "replay": {"max_wall_time_s": 3}}


class Robot:
    def __init__(self, config, state, clock):
        self.config = config
        self.state = np.array(state, dtype=float)
        self.clock = clock
        self.commands = []
        self.connected = self.stopped = self.closed = self.held = False
        self.connect_kwargs = None
        self.lag_ticks = 0
        self.always_lag = False
        self.command_duration = 0.0
        self.hard_error = 0.0
        self.close_error = False

    def connect(self, **kwargs):
        self.connect_kwargs = kwargs
        if np.any(np.abs(self.state - kwargs["expected_state"]) > kwargs["expected_tolerance"]):
            raise RuntimeError("pre-enable pose mismatch")
        self.connected = True

    def command(self, target):
        self.commands.append((self.clock.monotonic(), np.array(target)))
        self.clock.sleep(self.command_duration)
        if self.lag_ticks:
            self.lag_ticks -= 1
        elif not self.always_lag:
            self.state = np.array(target)
        return np.array(target)

    def read_state(self):
        result = self.state.copy()
        if self.commands:
            result[0] += self.hard_error
        return result

    def hold(self):
        self.held = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True
        if self.close_error:
            raise RuntimeError("close failed")


def run(plan, tmp_path, configure=lambda robot: None, **kwargs):
    clock = Clock()
    robots = []

    def factory(config):
        robot = Robot(config, plan["expected_initial_state"], clock)
        configure(robot)
        robots.append(robot)
        return robot

    result = execute_replay(plan, factory, tmp_path, monotonic=clock.monotonic,
                            sleep=clock.sleep, wall_time=clock.monotonic, **kwargs)
    return result, robots[0], clock


def test_replay_rate_velocity_start_gate_and_normal_hold(plan, tmp_path):
    report, robot, _ = run(plan, tmp_path)
    assert report["success"] and report["completed_actions"] == 2
    assert robot.closed and robot.held and not robot.stopped
    np.testing.assert_allclose(robot.connect_kwargs["expected_state"], plan["expected_initial_state"])
    np.testing.assert_allclose(robot.connect_kwargs["expected_tolerance"], plan["expected_initial_tolerance"])
    times = np.array([row[0] for row in robot.commands])
    assert np.all(np.diff(times) >= 1 / 30 - 1e-10)
    commands = np.array([row[1] for row in robot.commands])
    assert np.all(np.abs(np.diff(commands, axis=0))[:, :6] <= math.radians(5) / 30 + 1e-10)
    np.testing.assert_allclose(commands[-1], plan["actions"][-1])
    assert json.loads((tmp_path / "summary.json").read_text())["success"]


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(actions=[]),
    lambda p: p["actions"][0].__setitem__(0, float("nan")),
    lambda p: p["actions"][0].__setitem__(2, .5),
    lambda p: p["robot"].update(actuation_enabled=False),
    lambda p: p["replay"].update(watchdog_timeout_s=.01),
    lambda p: p.update(expected_initial_tolerance=[1] * 7),
    lambda p: p["robot"].update(control_hz=10),
    lambda p: p["replay"].update(joint_speed_rad_s=1.0),
])
def test_invalid_plan_fails_before_factory(plan, tmp_path, mutation):
    mutation(plan)
    created = []
    with pytest.raises((ValueError, TypeError)):
        execute_replay(plan, lambda c: created.append(c), tmp_path)
    assert not created


def test_lag_holds_interpolation_progress_and_then_recovers(plan, tmp_path):
    plan["actions"] = [[.05, .1, -.1, 0, .1, 0, .02]]
    plan["replay"].update(tracking_tolerance_rad=.001)
    report, robot, _ = run(plan, tmp_path, lambda r: setattr(r, "lag_ticks", 3))
    assert report["success"]
    records = [json.loads(line) for line in (tmp_path / "replay.jsonl").read_text().splitlines()]
    assert all(row["tracking_wait"] for row in records[:3])
    assert len({row["fraction"] for row in records[:4]}) == 1
    assert not robot.stopped


def test_permanent_lag_stops_and_records_failure(plan, tmp_path):
    plan["actions"] = [[.05, .1, -.1, 0, .1, 0, .02]]
    plan["replay"].update(tracking_tolerance_rad=.001, tracking_timeout_s=.1)
    robots = []
    def configure(robot):
        robot.always_lag = True
        robots.append(robot)
    with pytest.raises(ReplayError, match="tracking timeout"):
        run(plan, tmp_path, configure)
    assert robots[0].stopped and robots[0].closed and not robots[0].held
    assert not json.loads((tmp_path / "summary.json").read_text())["success"]


@pytest.mark.parametrize("setting,value,match", [
    ("hard_error", .3, "hard joint/gripper bound"),
    ("command_duration", .6, "heartbeat timed out"),
    ("close_error", True, "Close failed"),
])
def test_fault_paths(plan, tmp_path, setting, value, match):
    robots = []
    def configure(robot):
        setattr(robot, setting, value)
        robots.append(robot)
    with pytest.raises(ReplayError, match=match):
        run(plan, tmp_path, configure)
    assert robots[0].closed
    assert robots[0].stopped
    assert not json.loads((tmp_path / "summary.json").read_text())["success"]


def test_stop_file_prevents_connection(plan, tmp_path):
    stop = tmp_path / "STOP"
    stop.touch()
    with pytest.raises(ReplayError, match="before connection"):
        execute_replay(plan, lambda c: pytest.fail("must not create robot"), tmp_path, stop_file=stop)


def test_mid_replay_cancel_stops(plan, tmp_path):
    event = threading.Event()
    robots = []
    def configure(robot):
        original = robot.command
        def command(target):
            value = original(target)
            event.set()
            return value
        robot.command = command
        robots.append(robot)
    with pytest.raises(ReplayError, match="cancelled"):
        run(plan, tmp_path, configure, cancel=event)
    assert robots[0].stopped and robots[0].closed


def test_wall_time_is_finite(plan, tmp_path):
    plan["actions"][-1][0] = 1.0
    plan["replay"]["max_wall_time_s"] = .1
    with pytest.raises(ReplayError, match="wall-time limit"):
        run(plan, tmp_path)


def test_initial_mismatch_fails_before_enabled(plan, tmp_path):
    robots = []
    def configure(robot):
        robot.state[0] += .1
        robots.append(robot)
    with pytest.raises(ReplayError, match="pre-enable pose mismatch"):
        run(plan, tmp_path, configure)
    assert not robots[0].connected and not robots[0].commands
    assert robots[0].closed


def test_startup_clips_only_derived_approach_targets(plan, tmp_path):
    plan["expected_initial_state"][1] = math.radians(-1)
    plan["robot"]["joint_startup_recovery_tolerance_rad"] = [0, math.radians(3), 0, 0, 0, 0]
    report, robot, _ = run(plan, tmp_path)
    assert report["success"]
    assert all(target[1] >= 0 for _, target in robot.commands)
    records = [json.loads(line) for line in (tmp_path / "replay.jsonl").read_text().splitlines()]
    assert records[0]["interpolated_target"][1] < 0
    assert records[0]["target"][1] == 0


def test_watchdog_independently_stops_when_caller_blocks():
    stopped = threading.Event()
    class BlockedRobot:
        def stop(self):
            stopped.set()
    watchdog = ReplayWatchdog(BlockedRobot(), .05, time.monotonic() + 1, threading.Event(), None)
    watchdog.start()
    try:
        assert stopped.wait(.5)
        assert "heartbeat" in watchdog.failure
    finally:
        watchdog.close()


def test_cli_dryrun_does_not_import_hardware(plan, tmp_path):
    root = Path(__file__).resolve().parents[1]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    code = f'''import runpy, sys
sys.argv = ["replay_real_robot.py", "--plan", {str(path)!r}]
runpy.run_path({str(root / "scripts/replay_real_robot.py")!r}, run_name="inspect_module")
import importlib.util
spec = importlib.util.spec_from_file_location("replay_cli", {str(root / "scripts/replay_real_robot.py")!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.main() == 0
assert "pressb.robots.piper.real" not in sys.modules
assert "piper_sdk" not in sys.modules
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not json.loads(result.stdout)["executed"]


def test_mutated_validated_plan_rechecked_before_factory(plan, tmp_path):
    validated = ReplayPlan.from_mapping(plan)
    validated.actions[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        execute_replay(validated, lambda _: pytest.fail("must not create hardware"), tmp_path)


def test_effective_hardware_cap_matches_reviewed_replay_speed(plan, tmp_path):
    _, robot, _ = run(plan, tmp_path)
    np.testing.assert_allclose(robot.config["max_joint_velocity_rad_s"], math.radians(5))
    assert robot.config["max_gripper_velocity_m_s"] == .01
    assert isinstance(robot.connect_kwargs["cancel_event"], threading.Event)


def test_cancel_during_connect_is_passed_to_pre_enable_gate(plan, tmp_path):
    clock = Clock()
    cancelled = threading.Event()
    robots = []
    class ConnectingRobot(Robot):
        def connect(self, **kwargs):
            cancelled.set()
            assert kwargs["cancel_event"].is_set()
            raise RuntimeError("cancelled before enable")
    def factory(config):
        robot = ConnectingRobot(config, plan["expected_initial_state"], clock)
        robots.append(robot)
        return robot
    with pytest.raises(ReplayError, match="cancelled before enable"):
        execute_replay(plan, factory, tmp_path, cancel=cancelled,
                       monotonic=clock.monotonic, sleep=clock.sleep)
    assert not robots[0].connected and not robots[0].commands
    assert robots[0].closed


def test_close_deadline_cannot_report_success(plan, tmp_path):
    robots = []
    def configure(robot):
        original = robot.close
        def close():
            robot.clock.sleep(10)
            original()
        robot.close = close
        robots.append(robot)
    with pytest.raises(ReplayError, match="wall-time limit"):
        run(plan, tmp_path, configure)
    assert robots[0].closed
    assert not json.loads((tmp_path / "summary.json").read_text())["success"]


@pytest.fixture
def native_plan(plan):
    plan["expected_initial_state"][1] = math.radians(-1)
    plan["robot"]["joint_startup_recovery_tolerance_rad"] = [0, math.radians(3), 0, 0, 0, 0]
    plan["native_startup_recovery"] = {"verified_max_joint_speed_rad_s": [3.0] * 6}
    return plan


def install_native_recovery(robot):
    robot.native_kwargs = None
    robot.native_final_state = None

    def recover(**kwargs):
        robot.native_kwargs = kwargs
        initial = robot.state.copy()
        target = initial.copy()
        target[1] = kwargs["inset_rad"]
        started = robot.clock.monotonic()
        for index in range(10):
            robot.clock.sleep(.1)
            robot.state = initial + (target - initial) * min((index + 1) / 4, 1)
            # Within native tolerance, but intentionally distinct from its goal:
            # the runner must use fresh measured state for the normal approach.
            if index >= 4:
                robot.state[1] -= .001
            kwargs["progress_callback"]({
                "state": robot.state.tolist(), "target": target.tolist(),
                "elapsed_s": robot.clock.monotonic() - started,
                "stable_elapsed_s": max(0.0, (index - 3) * .1),
                "recovery_joint_indices": [1],
                "measured_speed_rad_s": None if index < 2 else [0, .03, 0, 0, 0, 0],
                "feedback_timestamps": {"joint_12": robot.clock.monotonic()},
            })
        robot.native_final_state = robot.state.copy()
        return {"completed": True, "initial_state": initial.tolist(), "target": target.tolist(),
                "final_state": robot.state.tolist(), "samples": 10}
    robot.recover_startup_native = recover


def test_native_recovery_opt_in_logs_and_uses_fresh_start(native_plan, tmp_path):
    report, robot, _ = run(native_plan, tmp_path, install_native_recovery)
    assert report["success"] and report["native_recovery_completed"]
    assert report["native_recovery_feedback_samples"] == 10
    assert robot.config["startup_recovery_timeout_s"] >= 20
    assert robot.native_kwargs["speed_percent"] == 1
    assert robot.native_kwargs["check_inward_progress"] is True
    assert "target" not in robot.native_kwargs
    assert isinstance(robot.native_kwargs["cancel_event"], threading.Event)
    np.testing.assert_allclose(report["post_native_recovery_state"], robot.native_final_state)
    records = [json.loads(line) for line in (tmp_path / "replay.jsonl").read_text().splitlines()]
    assert [row["phase"] for row in records[:10]] == ["native_recovery"] * 10
    assert records[10]["phase"] == "approach"
    assert records[0]["measured_speed_rad_s"] is None
    assert records[2]["measured_speed_rad_s"] == [0, .03, 0, 0, 0, 0]
    fraction = records[10]["fraction"]
    expected = robot.native_final_state + (np.array(native_plan["actions"][0]) - robot.native_final_state) * fraction
    np.testing.assert_allclose(robot.commands[0][1], expected)
    assert robot.closed and robot.held and not robot.stopped


def test_native_progress_policy_round_trips_and_reaches_driver(native_plan, tmp_path):
    native_plan["native_startup_recovery"]["check_inward_progress"] = False
    validated = ReplayPlan.from_mapping(native_plan)
    options = validated.native_startup_recovery.as_dict()
    assert options["check_inward_progress"] is False
    summary = json.loads(json.dumps(validated.summary(), allow_nan=False))
    assert summary["native_startup_recovery"]["check_inward_progress"] is False
    native_plan["native_startup_recovery"] = json.loads(json.dumps(options))
    report, robot, _ = run(native_plan, tmp_path, install_native_recovery)
    assert report["success"] and report["native_recovery_completed"]
    assert robot.native_kwargs["check_inward_progress"] is False
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["native_startup_recovery"]["check_inward_progress"] is False
    records = [json.loads(line) for line in (tmp_path / "replay.jsonl").read_text().splitlines()]
    native_records = [row for row in records if row["phase"] == "native_recovery"]
    assert len(native_records) == 10
    assert all(row["check_inward_progress"] is False for row in native_records)


@pytest.mark.parametrize("value", [None, 0, 1, "false", [], {}])
def test_native_progress_policy_rejects_nonboolean_before_hardware(native_plan, tmp_path, value):
    native_plan["native_startup_recovery"]["check_inward_progress"] = value
    with pytest.raises(ValueError, match="check_inward_progress"):
        execute_replay(native_plan, lambda _: pytest.fail("must not construct hardware"), tmp_path)


def test_native_mutated_progress_policy_revalidated_before_hardware(native_plan, tmp_path):
    validated = ReplayPlan.from_mapping(native_plan)
    object.__setattr__(validated.native_startup_recovery, "check_inward_progress", 0)
    with pytest.raises(ValueError, match="check_inward_progress"):
        execute_replay(validated, lambda _: pytest.fail("must not construct hardware"), tmp_path)


def test_native_disabled_by_default(plan, tmp_path):
    def configure(robot):
        robot.recover_startup_native = lambda **kwargs: pytest.fail("native recovery must be opt-in")
    report, _, _ = run(plan, tmp_path, configure)
    assert "native_startup_recovery" not in report
    assert "native_recovery_completed" not in report


@pytest.mark.parametrize("options", [
    None, {}, {"verified_max_joint_speed_rad_s": [3] * 6, "target": [0] * 7},
    {"verified_max_joint_speed_rad_s": [float("nan")] * 6},
    {"verified_max_joint_speed_rad_s": [0] * 6},
    {"verified_max_joint_speed_rad_s": [10] * 6},
    {"verified_max_joint_speed_rad_s": [3] * 6, "speed_percent": True},
    {"verified_max_joint_speed_rad_s": [3] * 6, "speed_percent": 2},
    {"verified_max_joint_speed_rad_s": [3] * 6, "timeout_s": 21},
    {"verified_max_joint_speed_rad_s": [3] * 6, "timeout_s": float("inf")},
    {"verified_max_joint_speed_rad_s": [3] * 6, "stable_s": .49},
    {"verified_max_joint_speed_rad_s": [3] * 6, "stable_s": 20},
    {"verified_max_joint_speed_rad_s": [3] * 6, "inset_rad": math.radians(1.1)},
    {"verified_max_joint_speed_rad_s": [3] * 6, "max_travel_rad": math.radians(4.1)},
    {"verified_max_joint_speed_rad_s": [3] * 6, "measured_speed_limit_rad_s": math.radians(5.1)},
    {"verified_max_joint_speed_rad_s": [3] * 6, "nonrecover_drift_rad": math.radians(1.1)},
    {"verified_max_joint_speed_rad_s": [3] * 6, "tolerance_rad": math.radians(.26)},
    {"verified_max_joint_speed_rad_s": [3] * 6, "inset_rad": math.radians(.25)},
    {"verified_max_joint_speed_rad_s": [3] * 6, "max_travel_rad": math.radians(1)},
])
def test_native_invalid_options_rejected_before_hardware(native_plan, tmp_path, options):
    native_plan["native_startup_recovery"] = options
    with pytest.raises(ValueError):
        execute_replay(native_plan, lambda _: pytest.fail("must not construct hardware"), tmp_path)


def test_native_rejects_other_joint_outside_bounds_before_hardware(native_plan, tmp_path):
    native_plan["expected_initial_state"][0] = math.radians(-155)
    native_plan["robot"]["joint_startup_recovery_tolerance_rad"][0] = math.radians(3)
    with pytest.raises(ValueError, match="only joints 2 and 3"):
        execute_replay(native_plan, lambda _: pytest.fail("must not construct hardware"), tmp_path)


def test_native_unsupported_driver_rejected_before_connect(native_plan, tmp_path):
    robots = []
    with pytest.raises(ReplayError, match="does not support"):
        run(native_plan, tmp_path, lambda r: robots.append(r))
    assert not robots[0].connected and not robots[0].commands
    assert robots[0].stopped and robots[0].closed


@pytest.mark.parametrize("mode,match", [
    ("error", "native hardware failure"),
    ("incomplete", "did not confirm completion"),
    ("bad_evidence", "Out of range float values"),
    ("stale", "heartbeat timed out"),
    ("outside", "outside nominal joint limits"),
    ("bad_target", "target outside nominal limits"),
    ("nan_feedback", "must be finite"),
])
def test_native_failure_never_starts_approach_and_stops(native_plan, tmp_path, mode, match):
    robots = []
    def configure(robot):
        robots.append(robot)
        def recover(**kwargs):
            if mode == "error":
                raise RuntimeError("native hardware failure")
            if mode == "incomplete":
                return {"completed": False}
            if mode == "bad_evidence":
                return {"completed": True, "elapsed_s": float("nan")}
            if mode == "stale":
                robot.clock.sleep(.6)
            if mode == "outside":
                return {"completed": True}
            target = robot.state.copy()
            target[1] = math.radians(1)
            state = robot.state.copy()
            if mode == "bad_target":
                target[2] = 1
            if mode == "nan_feedback":
                state[0] = float("nan")
            kwargs["progress_callback"]({"state": state.tolist(), "target": target.tolist()})
            return {"completed": True}
        robot.recover_startup_native = recover
    with pytest.raises(ReplayError, match=match):
        run(native_plan, tmp_path, configure)
    robot = robots[0]
    assert not robot.commands and not robot.held
    assert robot.stopped and robot.closed
    assert not json.loads((tmp_path / "summary.json").read_text())["success"]


@pytest.mark.parametrize("kind", ["event", "stop_file"])
def test_native_cancellation_checked_before_heartbeat(native_plan, tmp_path, kind):
    event = threading.Event()
    stop_file = tmp_path / "STOP"
    robots = []
    def configure(robot):
        robots.append(robot)
        def recover(**kwargs):
            if kind == "event":
                event.set()
            else:
                stop_file.touch()
            kwargs["progress_callback"]({"state": robot.state.tolist(), "target": [0] * 7})
            pytest.fail("cancellation must interrupt callback")
        robot.recover_startup_native = recover
    with pytest.raises(ReplayError, match="cancelled"):
        run(native_plan, tmp_path, configure, cancel=event, stop_file=stop_file)
    assert robots[0].stopped and robots[0].closed and not robots[0].commands
    assert not (tmp_path / "replay.jsonl").read_text()


@pytest.mark.parametrize("global_limit", [False, True])
def test_native_timeout_with_regular_feedback(native_plan, tmp_path, global_limit):
    native_plan["native_startup_recovery"]["timeout_s"] = 1
    if global_limit:
        native_plan["replay"]["max_wall_time_s"] = .7
    robots = []
    def configure(robot):
        robots.append(robot)
        def recover(**kwargs):
            while True:
                robot.clock.sleep(.1)
                kwargs["progress_callback"]({"state": robot.state.tolist(), "target": [0] * 7})
        robot.recover_startup_native = recover
    match = "wall-time limit" if global_limit else "Native startup recovery time limit"
    with pytest.raises(ReplayError, match=match):
        run(native_plan, tmp_path, configure)
    assert robots[0].stopped and robots[0].closed and not robots[0].commands


def test_native_watchdog_stops_blocked_driver_independently(native_plan, tmp_path):
    native_plan["replay"]["watchdog_timeout_s"] = .05
    clock = type("RealClock", (), {"monotonic": staticmethod(time.monotonic), "sleep": staticmethod(time.sleep)})()
    stopped = threading.Event()
    robots = []
    def factory(config):
        robot = Robot(config, native_plan["expected_initial_state"], clock)
        original_stop = robot.stop
        def stop():
            original_stop()
            stopped.set()
        robot.stop = stop
        def recover(**kwargs):
            assert stopped.wait(.5), "independent watchdog must stop while driver blocks"
            return {"completed": True}
        robot.recover_startup_native = recover
        robots.append(robot)
        return robot
    with pytest.raises(ReplayError, match="heartbeat timed out"):
        execute_replay(native_plan, factory, tmp_path)
    assert robots[0].stopped and robots[0].closed and not robots[0].commands


def test_native_dryrun_explains_distinct_control_mode(native_plan):
    summary = ReplayPlan.from_mapping(native_plan).summary()
    native = summary["native_startup_recovery"]
    assert "firmware-native" in native["control_mode"]
    assert "not normal command slew" in native["control_mode"]
    assert "fresh measured state" in native["target_rule"]
    assert "target" not in native


def test_native_mutated_options_revalidated_before_factory(native_plan, tmp_path):
    validated = ReplayPlan.from_mapping(native_plan)
    object.__setattr__(validated.native_startup_recovery, "speed_percent", 100)
    with pytest.raises(ValueError, match="speed_percent=1"):
        execute_replay(validated, lambda _: pytest.fail("must not construct hardware"), tmp_path)


def test_native_nominal_noop_needs_no_callback_or_native_move(plan, tmp_path):
    plan["native_startup_recovery"] = {"verified_max_joint_speed_rad_s": [3] * 6}
    def configure(robot):
        def recover(**kwargs):
            return {"completed": True, "target": robot.state.tolist(), "state": robot.state.tolist()}
        robot.recover_startup_native = recover
    report, robot, _ = run(plan, tmp_path, configure)
    assert report["success"] and report["native_recovery_completed"]
    assert report["native_recovery_feedback_samples"] == 0
    assert robot.closed and not robot.stopped


def test_native_runner_integrates_piper_adapter_with_dynamic_can_fake(tmp_path):
    """Exercise firmware-native recovery then normal control across both layers."""
    from test_piper_real import Clock as HardwareClock, SDK
    from pressb.robots.piper.real import PiperHardware

    clock = HardwareClock()
    sdk = SDK(clock)
    sdk.values = np.array([0, -1507, 2381, 0, 0, 0, 0])
    goal = sdk.values.copy()
    raw_joint = sdk.JointCtrl
    raw_sleep = clock.sleep

    def set_goal(*values):
        raw_joint(*values)
        goal[:6] = values

    def advance(duration):
        # A device maximum of 3 rad/s at 1% gives 0.03 rad/s, below the
        # independently observed 5 degree/s guard. Update actual feedback only
        # as simulated time advances, never teleport at JointCtrl.
        max_raw_step = np.rad2deg(.03) * 1000 * duration
        sdk.values[:6] += np.rint(np.clip(goal[:6] - sdk.values[:6], -max_raw_step, max_raw_step)).astype(int)
        raw_sleep(duration)

    sdk.JointCtrl = set_goal
    clock.sleep = advance
    initial = np.r_[np.deg2rad(sdk.values[:6] / 1000), 0.0]
    plan = {"schema_version": 1, "source": {"dataset": "dynamic CAN fake", "episode_index": 0},
            "fps": 30, "actions": [[0.0] * 7], "expected_initial_state": initial.tolist(),
            "expected_initial_tolerance": [math.radians(1)] * 6 + [.001],
            "robot": {"actuation_enabled": True, "control_hz": 30,
                      "joint_startup_recovery_tolerance_rad": [0, math.radians(3), math.radians(3), 0, 0, 0]},
            "replay": {"max_wall_time_s": 30},
            "native_startup_recovery": {"verified_max_joint_speed_rad_s": [3.0] * 6}}
    hardware = []
    def factory(config):
        instance = PiperHardware(config, sdk_factory=lambda _: sdk, can_probe=lambda _: None,
                                 monotonic=clock.monotonic, wall_time=clock.wall, sleep=clock.sleep)
        hardware.append(instance)
        return instance
    report = execute_replay(plan, factory, tmp_path, monotonic=clock.monotonic,
                            sleep=clock.sleep, wall_time=clock.wall)
    assert report["success"] and report["native_recovery_completed"]
    assert report["native_recovery_feedback_samples"] > 25
    assert report["native_recovery_result"]["completed"]
    assert report["native_recovery_result"]["stable_elapsed_s"] >= .5
    joints = [call[1:] for call in sdk.calls if call[0] == "joints"]
    assert joints[0] == (0, 1000, -1000, 0, 0, 0)
    assert all(target[1] >= 0 and target[2] <= 0 for target in joints)
    assert not any(call[0] == "stop" for call in sdk.calls)
    assert sdk.calls[-1] == ("disconnect",)
    assert not hardware[0].status()["connected"]
