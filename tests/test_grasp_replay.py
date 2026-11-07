"""Synthetic contact/load evidence, including the observed 34.8 mm plateau."""
import json
import math
import threading
from dataclasses import replace

import numpy as np
import pytest

from pressb.control.grasp_replay import GraspFeedbackError, GraspMonitor, GraspPolicy, GraspTaskFailure
from pressb.control.real_replay import ReplayError, ReplayPlan, ReplayTaskFailure, execute_replay
from test_real_replay import Clock, Robot


@pytest.fixture
def grasp_plan():
    pose = [0, .1, -.1, 0, .1, 0]
    return {"schema_version": 1, "source": {"dataset": "synthetic loaded-contact fixture"}, "fps": 30,
            "actions": [pose + [.04], pose + [.029667], [.05] + pose[1:] + [.029667],
                        [.05] + pose[1:] + [.05]],
            "action_modes": ["position", "close", "carry", "release"],
            "expected_initial_state": pose + [.04],
            "expected_initial_tolerance": [math.radians(1)] * 6 + [.001],
            "robot": {"actuation_enabled": True, "control_hz": 30},
            "replay": {"max_wall_time_s": 20},
            "grasp_policy": {"expected_width_m": [.032, .038], "min_effort_raw": 800,
                             "close_timeout_s": 3}}


class ContactRobot(Robot):
    def __init__(self, *args):
        super().__init__(*args)
        self.plateau = .0348
        self.effort = 1040
        self.last_target = self.state.copy()
        self.feedback_calls = 0
        self.fault = False
        self.enabled = True
        self.frozen_stamp = None
        self.drop_load_after_lift = False
        self.stall_arm = False
        self.arm_offset_ticks = 0
        self.preserved_gripper_goal = None
        self.cleanup_targets = []

    def command(self, target):
        self.preserved_gripper_goal = None
        value = super().command(target)
        self.last_target = np.array(target)
        if self.plateau is not None:
            self.state[6] = max(target[6], self.plateau)
        if self.stall_arm:
            self.state[0] = 0
        if self.arm_offset_ticks > 0:
            self.state[0] += .01
            self.arm_offset_ticks -= 1
        return value

    def hold(self):
        self.held = True
        self.preserved_gripper_goal = None
        self.last_target = self.state.copy()
        self.cleanup_targets.append(self.last_target.copy())
        return self.last_target.copy()

    def hold_grasp(self, *, gripper_target_m):
        assert abs(gripper_target_m - self.last_target[6]) <= 1e-6
        self.preserved_gripper_goal = float(self.last_target[6])
        self.last_target = self.state.copy()
        self.last_target[6] = self.preserved_gripper_goal
        self.held = True
        self.cleanup_targets.append(self.last_target.copy())
        return self.last_target.copy()

    def close(self):
        if self.connected and not self.stopped:
            if self.preserved_gripper_goal is None:
                self.hold()
            else:
                self.hold_grasp(gripper_target_m=self.preserved_gripper_goal)
        super().close()

    def read_gripper_feedback(self):
        self.feedback_calls += 1
        if self.frozen_stamp is not None and hasattr(self, "frozen_packet"):
            return self.frozen_packet.copy()
        loaded = self.plateau is not None and self.last_target[6] < self.plateau - .0005
        effort = self.effort if loaded else 0
        if self.drop_load_after_lift and self.state[0] > .01:
            effort = 0
        packet = {"width_m": float(self.state[6]), "effort_raw": effort,
                "status_raw": (64 if self.enabled else 0) | (1 if self.fault else 0),
                "enabled": self.enabled, "fault": self.fault,
                "timestamp_s": self.clock.monotonic() if self.frozen_stamp is None else self.frozen_stamp}
        if self.frozen_stamp is not None:
            self.frozen_packet = packet.copy()
        return packet


def run_grasp(plan, path, configure=lambda _: None, **kwargs):
    clock = Clock()
    robot = ContactRobot(plan["robot"], plan["expected_initial_state"], clock)
    configure(robot)
    result = execute_replay(plan, lambda _: robot, path, monotonic=clock.monotonic,
                            sleep=clock.sleep, wall_time=clock.monotonic, **kwargs)
    return result, robot


def report(path):
    return json.loads((path / "summary.json").read_text())


def records(path):
    return [json.loads(line) for line in (path / "replay.jsonl").read_text().splitlines()]


def test_observed_loaded_plateau_acquires_carries_then_releases(grasp_plan, tmp_path):
    result, robot = run_grasp(grasp_plan, tmp_path)
    assert result["success"] and result["completed_actions"] == 4
    assert result["grasp"]["acquisitions"] == result["grasp"]["releases"] == 1
    assert result["grasp"]["outcome"] == "acquired_and_released"
    assert not result["grasp"]["acquired"]
    assert robot.held and robot.closed and not robot.stopped
    assert robot.feedback_calls == result["ticks"]
    rows = records(tmp_path)
    carry = [row for row in rows if row["action_mode"] == "carry"]
    assert carry and all(row["grasp"]["acquired"] for row in carry)
    assert all(row["target"][6] == pytest.approx(.029667) for row in carry)
    assert all(row["observed"][6] == pytest.approx(.0348) for row in carry)
    assert all(row["grasp"]["feedback"]["effort_raw"] == 1040 for row in carry)
    assert any(abs(row["observed"][6] - row["target"][6]) > .005 for row in carry)
    assert all(not row["grasp"]["soft_gripper_arrival_exempt"] for row in rows if row["action_mode"] == "release")
    assert result["final_state"][6] >= .039


def dense_release_plan(grasp_plan):
    """Reproduce the real 35.1 mm grasp / 28.167 mm first opening target."""
    pose = grasp_plan["actions"][0][:6]
    carried = [.05] + pose[1:]
    release = [carried + [float(width)] for width in np.linspace(.028, .06, 193)[1:]]
    grasp_plan["actions"] = [pose + [.04], pose + [.028], carried + [.028], *release]
    grasp_plan["action_modes"] = ["position", "close", "carry"] + ["release"] * len(release)
    grasp_plan["grasp_policy"].update(expected_width_m=[.029, .037], min_effort_raw=600,
                                     max_effort_raw=1500, close_timeout_s=6.)
    grasp_plan["replay"].update(tracking_tolerance_rad=math.radians(.5), max_wall_time_s=30)
    return grasp_plan


def test_dense_release_crosses_loaded_gap_then_restores_position_tracking(grasp_plan, tmp_path):
    plan = dense_release_plan(grasp_plan)
    result, robot = run_grasp(plan, tmp_path, lambda r: setattr(r, "plateau", .0351))
    rows = [r for r in records(tmp_path) if r["action_mode"] == "release"]
    assert result["success"] and not robot.stopped
    assert result["grasp"]["releases"] == 1
    assert rows[0]["target"][6] < .029 and rows[0]["observed"][6] == pytest.approx(.0351)
    assert rows[0]["grasp"]["release_gripper_arrival_exempt"]
    assert not rows[0]["tracking_wait"]
    assert all(not r["grasp"]["soft_gripper_arrival_exempt"] for r in rows)
    past_contact = [r for r in rows if r["target"][6] >= .0351]
    assert past_contact and all(not r["grasp"]["release_bridge_active"] for r in past_contact)
    assert all(not r["grasp"]["release_gripper_arrival_exempt"] for r in past_contact)
    assert result["final_state"][6] == pytest.approx(.06)


@pytest.mark.parametrize("case,match", [
    ("stuck_closed", "tracking timeout"), ("arm_lag", "tracking timeout"),
    ("hard_width", "hard joint/gripper bound"), ("stale", "stale"),
    ("fault", "fault or lost enable"), ("overload", "effort exceeds"),
])
def test_release_bridge_keeps_fault_tracking_and_freshness_guards(grasp_plan, tmp_path, case, match):
    plan = dense_release_plan(grasp_plan)
    robots = []
    def configure(robot):
        robot.plateau = .0351
        original = robot.command
        releasing = False
        def command(target):
            nonlocal releasing
            if robot.state[0] > .049 and target[6] > .028:
                releasing = True
            result = original(target)
            if releasing:
                if case == "stuck_closed":
                    robot.state[6] = .0351
                elif case == "arm_lag":
                    robot.state[0] += .02
                elif case == "hard_width":
                    robot.state[6] = .06
                elif case == "stale" and robot.frozen_stamp is None:
                    robot.frozen_stamp = robot.clock.monotonic()
                elif case == "fault":
                    robot.fault = True
                elif case == "overload":
                    robot.effort = 2000
            return result
        robot.command = command
        robots.append(robot)
    with pytest.raises(ReplayError, match=match):
        run_grasp(plan, tmp_path, configure)
    result = report(tmp_path)
    assert not result["success"] and result["failure_kind"] == "hardware_or_control"
    assert result["emergency_stop_requested"] and robots[0].stopped
    assert result["grasp"]["releases"] == 0
    assert result["grasp"]["mode"] == "release"  # Hard errors stop before logging the tick.


def test_release_bridge_deadline_spans_many_actions(grasp_plan, tmp_path):
    plan = dense_release_plan(grasp_plan)
    carried = plan["actions"][2][:6]
    plan["actions"] = plan["actions"][:3] + [carried + [.0281]] * 300 + [carried + [.06]]
    plan["action_modes"] = ["position", "close", "carry"] + ["release"] * 301
    with pytest.raises(ReplayError, match="Release opening bridge exceeded tracking timeout"):
        run_grasp(plan, tmp_path, lambda r: setattr(r, "plateau", .0351))
    result = report(tmp_path)
    assert 10 < result["completed_actions"] < 300
    assert result["grasp"]["releases"] == 0
    assert result["emergency_stop_requested"]


@pytest.mark.parametrize("case", ["empty", "wrong_width", "low_effort"])
def test_expected_contact_failures_hold_before_any_lift(grasp_plan, tmp_path, case):
    robots = []
    def configure(robot):
        if case == "empty":
            robot.plateau = None
        elif case == "wrong_width":
            robot.plateau = .0405  # Obstruction outside reviewed object interval, still inside hard 15 mm.
        else:
            robot.effort = 100
        robots.append(robot)
    with pytest.raises(ReplayTaskFailure, match="lift blocked"):
        run_grasp(grasp_plan, tmp_path, configure)
    robot = robots[0]
    assert robot.held and robot.closed and not robot.stopped
    assert all(abs(target[0]) < 1e-12 for _, target in robot.commands)
    result = report(tmp_path)
    assert not result["success"] and result["failure_kind"] == "task"
    assert result["task_outcome"] == "grasp_not_acquired"
    assert result["emergency_stop_requested"] is False
    assert result["grasp"]["acquisitions"] == 0


def test_carry_loss_holds_before_further_lift_or_release(grasp_plan, tmp_path):
    robots = []
    def configure(robot):
        robot.drop_load_after_lift = True
        robots.append(robot)
    with pytest.raises(ReplayTaskFailure, match="further motion blocked"):
        run_grasp(grasp_plan, tmp_path, configure)
    robot = robots[0]
    assert robot.held and robot.closed and not robot.stopped
    assert .01 < max(target[0] for _, target in robot.commands) < .05
    assert report(tmp_path)["task_outcome"] == "grasp_lost"
    assert report(tmp_path)["completed_actions"] == 2
    assert not any(row.get("action_mode") == "release" for row in records(tmp_path))
    assert len(robot.cleanup_targets) == 2  # Hold, then close's second hold.
    assert all(target[6] == pytest.approx(.029667) for target in robot.cleanup_targets)
    assert all(target[6] != pytest.approx(.0348) for target in robot.cleanup_targets)


@pytest.mark.parametrize("case,match", [
    ("fault", "fault or lost enable"), ("disabled", "fault or lost enable"),
    ("stale", "stale"), ("overload", "effort exceeds"), ("hard_gap", "hard joint/gripper bound"),
])
def test_hardware_failures_still_stop(grasp_plan, tmp_path, case, match):
    grasp_plan["grasp_policy"]["max_effort_raw"] = 2000
    robots = []
    def configure(robot):
        if case == "fault":
            robot.fault = True
        elif case == "disabled":
            robot.enabled = False
        elif case == "stale":
            robot.frozen_stamp = 0
        elif case == "overload":
            robot.effort = 3000
        else:
            robot.plateau = .06
        robots.append(robot)
    with pytest.raises(ReplayError, match=match) as error:
        run_grasp(grasp_plan, tmp_path, configure)
    assert not isinstance(error.value, ReplayTaskFailure)
    assert robots[0].stopped and robots[0].closed
    assert report(tmp_path)["failure_kind"] == "hardware_or_control"


def test_close_waits_for_tight_arm_pose_before_closing(grasp_plan, tmp_path):
    result, robot = run_grasp(grasp_plan, tmp_path, lambda r: setattr(r, "arm_offset_ticks", 4))
    assert result["success"]
    closing = [row for row in records(tmp_path) if row["action_mode"] == "close"]
    assert len(closing) > 4
    assert all(row["target"][6] == pytest.approx(.04) for row in closing[:4])
    assert closing[4]["target"][6] < .04


def test_grasp_mode_is_preserved_when_execute_revalidates_plan_object(grasp_plan, tmp_path):
    validated = ReplayPlan.from_mapping(grasp_plan)
    clock = Clock()
    robot = ContactRobot(grasp_plan["robot"], grasp_plan["expected_initial_state"], clock)
    result = execute_replay(validated, lambda _: robot, tmp_path, monotonic=clock.monotonic,
                            sleep=clock.sleep, wall_time=clock.monotonic)
    assert result["grasp"]["acquisitions"] == 1
    assert result["action_modes"] == grasp_plan["action_modes"]


@pytest.mark.parametrize("mutation", [
    lambda p: p.pop("action_modes"),
    lambda p: p.pop("grasp_policy"),
    lambda p: p["action_modes"].__setitem__(0, "close"),
    lambda p: p["action_modes"].__setitem__(1, "carry"),
    lambda p: p["action_modes"].__setitem__(2, "position"),
    lambda p: p["action_modes"].__setitem__(3, "carry"),
    lambda p: p["action_modes"].__setitem__(1, "invalid"),
    lambda p: p["action_modes"].pop(),
    lambda p: p["actions"][1].__setitem__(0, .1),
    lambda p: p["actions"][1].__setitem__(6, .041),
    lambda p: p["actions"][2].__setitem__(6, .03),
    lambda p: p["actions"][3].__setitem__(6, .035),
    lambda p: p["actions"][3].__setitem__(0, .1),
    lambda p: p["replay"].update(max_tracking_error_gripper_m=.016),
])
def test_invalid_phase_program_rejected_before_robot(grasp_plan, tmp_path, mutation):
    mutation(grasp_plan)
    with pytest.raises(ValueError):
        execute_replay(grasp_plan, lambda _: pytest.fail("must reject before robot construction"), tmp_path)


@pytest.mark.parametrize("field,value", [
    ("unknown", 1), ("expected_width_m", [float("nan"), .04]),
    ("expected_width_m", [0, .04]), ("expected_width_m", [.04, .03]),
    ("min_effort_raw", 0), ("min_effort_raw", True), ("max_effort_raw", 32768),
    ("max_effort_raw", 799), ("contact_dwell_s", .1), ("contact_dwell_s", float("inf")),
    ("carry_loss_grace_s", .5), ("max_feedback_age_s", .3), ("close_timeout_s", 31),
    ("max_width_speed_m_s", .1), ("min_closing_gap_m", 0),
    ("arm_stationary_tolerance_rad", .1), ("arm_tracking_tolerance_rad", .1),
])
def test_invalid_grasp_policy_rejected_before_robot(grasp_plan, tmp_path, field, value):
    grasp_plan["grasp_policy"][field] = value
    with pytest.raises(ValueError):
        execute_replay(grasp_plan, lambda _: pytest.fail("must reject before robot construction"), tmp_path)


def test_missing_gripper_api_rejected_before_enable(grasp_plan, tmp_path):
    clock = Clock()
    robot = Robot(grasp_plan["robot"], grasp_plan["expected_initial_state"], clock)
    with pytest.raises(ReplayError, match="requires read_gripper_feedback"):
        execute_replay(grasp_plan, lambda _: robot, tmp_path)
    assert not robot.connected and not robot.commands


def feedback(stamp, width=.0348, effort=1040, status=64):
    return {"width_m": width, "effort_raw": effort, "status_raw": status,
            "enabled": bool(status & 64), "fault": bool(status & 63), "timestamp_s": stamp}


def monitor(grasp_plan):
    plan = ReplayPlan.from_mapping(grasp_plan)
    result = GraspMonitor(plan.grasp_policy, plan.action_modes, 0, .07)
    result.begin_action(1, 0)
    return result


def observe(mon, stamp, *, now=None, **kwargs):
    mon.observe(feedback(stamp, **kwargs), arm_ready=True, applied_gripper=.029667,
                now=stamp if now is None else now, wall_time=stamp if now is None else now)


def test_duplicate_frames_never_accumulate_contact_dwell(grasp_plan):
    mon = monitor(grasp_plan)
    observe(mon, 0)
    observe(mon, .03)
    for _ in range(30):
        observe(mon, .03, now=.14)
    assert not mon.acquired and mon.acquisitions == 0
    assert mon.candidate_started == .03
    with pytest.raises(GraspFeedbackError, match="stale"):
        observe(mon, .03, now=.3)


def test_signed_effort_absolute_load_and_stable_distinct_frames(grasp_plan):
    mon = monitor(grasp_plan)
    for stamp in np.arange(0, .45, .03):
        observe(mon, float(stamp), effort=-1040)
    assert mon.acquired and mon.acquisitions == 1


def test_moving_width_does_not_count_as_loaded_contact(grasp_plan):
    mon = monitor(grasp_plan)
    for index, stamp in enumerate(np.arange(0, .9, .03)):
        observe(mon, float(stamp), width=.034 + (index % 2) * .001)
    assert not mon.acquired


def test_short_carry_load_dip_recovers_without_losing_grasp(grasp_plan):
    mon = monitor(grasp_plan)
    for stamp in np.arange(0, .45, .03):
        observe(mon, float(stamp))
    mon.begin_action(2, .45)
    observe(mon, .45, effort=0)
    observe(mon, .50, effort=0)
    observe(mon, .55, effort=1040)
    assert mon.acquired and mon.loss_started is None


def test_release_uses_last_trusted_contact_not_anomaly_during_loss_grace(grasp_plan):
    mon = monitor(grasp_plan)
    for stamp in np.arange(0, .45, .03):
        observe(mon, float(stamp))
    mon.begin_action(2, .45)
    observe(mon, .45, width=.04, effort=0)
    assert mon.acquired and not mon.contact_valid
    mon.begin_action(3, .46)
    observe(mon, .46, width=.04, effort=0)
    assert mon.release_contact_width_m == pytest.approx(.0348)
    assert not mon.release_soft_gripper_exempt(.03, .04)
    assert not mon.soft_gripper_exempt


@pytest.mark.parametrize("target,measured", [(.0348, .0348), (.0347, .0347)])
def test_release_bridge_retires_without_reenabling_when_width_changes(grasp_plan, target, measured):
    mon = monitor(grasp_plan)
    for stamp in np.arange(0, .45, .03):
        observe(mon, float(stamp))
    mon.begin_action(3, .45)
    observe(mon, .45)
    assert mon.release_soft_gripper_exempt(.03, .0348)
    assert not mon.release_soft_gripper_exempt(target, measured)
    assert not mon.release_bridge_active
    assert not mon.release_soft_gripper_exempt(target, .037)


def test_carry_before_acquisition_blocked_at_runtime(grasp_plan):
    mon = monitor(grasp_plan)
    with pytest.raises(GraspTaskFailure, match="no acquired grasp"):
        mon.begin_action(2, 0)


@pytest.mark.parametrize("mutation,match", [
    (lambda f: f.update(effort_raw=40000), "int16"),
    (lambda f: f.update(status_raw=300), "uint8"),
    (lambda f: f.update(width_m=float("nan")), "finite"),
    (lambda f: f.update(enabled=False), "disagree"),
    (lambda f: f.pop("timestamp_s"), "missing"),
])
def test_malformed_gripper_feedback_is_hardware_error(grasp_plan, mutation, match):
    mon = monitor(grasp_plan)
    data = feedback(0)
    mutation(data)
    with pytest.raises(GraspFeedbackError, match=match):
        mon.observe(data, arm_ready=True, applied_gripper=.029667, now=0, wall_time=0)


def test_same_timestamp_with_different_frame_contents_rejected(grasp_plan):
    mon = monitor(grasp_plan)
    observe(mon, .1)
    with pytest.raises(GraspFeedbackError, match="without a new frame"):
        observe(mon, .1, effort=1041)


def test_homed_status_bit_is_not_treated_as_fault(grasp_plan):
    mon = monitor(grasp_plan)
    observe(mon, 0, status=192)
    assert mon.feedback["enabled"] and not mon.feedback["fault"]


def test_task_failure_hold_fault_escalates_to_stop(grasp_plan, tmp_path):
    robots = []
    def configure(robot):
        robot.plateau = None
        def failed_hold():
            raise RuntimeError("hold feedback lost")
        robot.hold = failed_hold
        robots.append(robot)
    with pytest.raises(ReplayError) as error:
        run_grasp(grasp_plan, tmp_path, configure)
    assert not isinstance(error.value, ReplayTaskFailure)
    assert robots[0].stopped and robots[0].closed
    assert report(tmp_path)["emergency_stop_requested"]


def test_grasp_does_not_override_cancel_stop_path(grasp_plan, tmp_path):
    event = threading.Event()
    robots = []
    def configure(robot):
        original = robot.read_gripper_feedback
        def read():
            if robot.last_target[6] < .034:
                event.set()
            return original()
        robot.read_gripper_feedback = read
        robots.append(robot)
    with pytest.raises(ReplayError, match="cancelled") as error:
        run_grasp(grasp_plan, tmp_path, configure, cancel=event)
    assert not isinstance(error.value, ReplayTaskFailure)
    assert robots[0].stopped and robots[0].closed


def test_without_grasp_policy_legacy_position_timeout_is_unchanged(grasp_plan, tmp_path):
    grasp_plan.pop("action_modes")
    grasp_plan.pop("grasp_policy")
    robots = []
    def configure(robot):
        robots.append(robot)
        robot.read_gripper_feedback = lambda: pytest.fail("legacy plans must not read load feedback")
    with pytest.raises(ReplayError, match="tracking timeout"):
        run_grasp(grasp_plan, tmp_path, configure)
    assert robots[0].stopped and robots[0].closed
    assert "grasp" not in report(tmp_path)


def test_arm_tracking_fault_is_not_misreported_as_empty_grasp(grasp_plan, tmp_path):
    robots = []
    def configure(robot):
        robot.arm_offset_ticks = 1000  # 0.01 rad > grasp tolerance, but below hard bound.
        robots.append(robot)
    with pytest.raises(ReplayError, match="tracking timeout") as error:
        run_grasp(grasp_plan, tmp_path, configure)
    assert not isinstance(error.value, ReplayTaskFailure)
    assert robots[0].stopped and not robots[0].held
    assert report(tmp_path)["grasp"]["close_active_s"] == 0


@pytest.mark.parametrize("field", ["grasp_policy", "action_modes"])
def test_revalidation_rejects_partially_removed_grasp_contract(grasp_plan, tmp_path, field):
    altered = replace(ReplayPlan.from_mapping(grasp_plan), **{field: None})
    with pytest.raises(ValueError, match="both be absent or both be present"):
        execute_replay(altered, lambda _: pytest.fail("must reject before hardware"), tmp_path)


def test_release_whole_block_stays_stationary(grasp_plan, tmp_path):
    grasp_plan["actions"].append(list(grasp_plan["actions"][-1]))
    grasp_plan["actions"][-1][0] += .05
    grasp_plan["action_modes"].append("release")
    with pytest.raises(ValueError, match="stationary throughout release"):
        execute_replay(grasp_plan, lambda _: pytest.fail("must reject before hardware"), tmp_path)


def test_grasp_arm_tolerance_does_not_relax_stricter_replay_tracking(grasp_plan, tmp_path):
    grasp_plan["replay"]["tracking_tolerance_rad"] = .005
    robots = []
    def configure(robot):
        original = robot.command
        def command(target):
            result = original(target)
            # Position arrives normally; then close has an error below policy's
            # 0.5 degrees but above the stricter replay's 0.005 radians.
            if target[6] < .04:
                robot.state[0] += .006
            return result
        robot.command = command
        robots.append(robot)
    with pytest.raises(ReplayError, match="tracking timeout"):
        run_grasp(grasp_plan, tmp_path, configure)
    assert robots[0].stopped
    assert not report(tmp_path)["grasp"]["acquired"]


def test_preserving_hold_capability_required_before_enable(grasp_plan, tmp_path):
    robots = []
    def configure(robot):
        robot.hold_grasp = None
        robots.append(robot)
    with pytest.raises(ReplayError, match="requires hold_grasp"):
        run_grasp(grasp_plan, tmp_path, configure)
    assert not robots[0].connected and not robots[0].commands


def test_fresh_gripper_frame_retains_hard_bound_before_task_classification(grasp_plan):
    mon = monitor(grasp_plan)
    with pytest.raises(GraspFeedbackError, match="hard joint/gripper bound"):
        observe(mon, 0, width=.06)


def test_transient_unsettled_arm_resets_contact_dwell_even_on_duplicate_frame(grasp_plan):
    mon = monitor(grasp_plan)
    for stamp in (0., .05, .1, .15, .2, .25):
        observe(mon, stamp)
    mon.observe(feedback(.25), arm_ready=False, applied_gripper=.029667, now=.26, wall_time=.26)
    assert mon.candidate_started is None
    observe(mon, .35)
    assert not mon.acquired
