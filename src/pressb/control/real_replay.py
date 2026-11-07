"""Explicit, finite replay of absolute joint targets on one physical robot.

Plan inspection is hardware-independent. Execution requires a robot adapter whose
``connect(expected_state=..., expected_tolerance=...)`` checks fresh feedback
*before* enabling motors. This is an application watchdog, not a hard real-time
emergency-stop mechanism: a blocked SDK/kernel call can delay the stop request.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import json
import math
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping

import numpy as np

from pressb.control.grasp_replay import (
    GraspMonitor, GraspPolicy, GraspTaskFailure, validate_grasp_program,
)
from pressb.control.reference_execution import (
    SOURCE as REFERENCE_SOURCE, MAX_DELTA as REFERENCE_MAX_DELTA,
    SETTLE_TIME_S as REFERENCE_SETTLE_TIME_S, reference_limits, reference_target,
)


class ReplayError(RuntimeError):
    pass


class ReplayTaskFailure(ReplayError):
    """Replay held and ended after a task failure, without an emergency stop."""


def _vector(value, name, length=7):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or array.shape != (length,):
        raise ValueError(f"{name} must contain {length} real numbers")
    result = array.astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value, name):
    if isinstance(value, (bool, str)):
        raise ValueError(f"{name} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


@dataclass(frozen=True)
class ReplayOptions:
    control_hz: float = 30.0
    joint_speed_rad_s: float = math.radians(5)
    gripper_speed_m_s: float = 0.01
    max_wall_time_s: float = 120.0
    tracking_tolerance_rad: float = math.radians(2)
    tracking_tolerance_gripper_m: float = 0.005
    max_tracking_error_rad: float = math.radians(10)
    max_tracking_error_gripper_m: float = 0.015
    tracking_timeout_s: float = 5.0
    watchdog_timeout_s: float = 0.5
    close_timeout_s: float = 5.0
    execution_mode: str = "tracked"
    reference_chunk_size: int | None = None

    def __post_init__(self):
        if type(self.execution_mode) is not str or self.execution_mode not in ("tracked", "reference"):
            raise ValueError("execution_mode must be tracked or reference")
        if self.reference_chunk_size is not None and (
                type(self.reference_chunk_size) is not int or self.reference_chunk_size <= 0):
            raise ValueError("reference_chunk_size must be a positive integer or null")
        for field in fields(self):
            if field.name in ("execution_mode", "reference_chunk_size"):
                continue
            object.__setattr__(self, field.name, _positive(getattr(self, field.name), field.name))
        if self.tracking_tolerance_rad >= self.max_tracking_error_rad:
            raise ValueError("Arm tracking tolerance must be below hard tracking bound")
        if self.tracking_tolerance_gripper_m >= self.max_tracking_error_gripper_m:
            raise ValueError("Gripper tracking tolerance must be below hard tracking bound")
        if self.watchdog_timeout_s <= 1.0 / self.control_hz:
            raise ValueError("Watchdog timeout must exceed one control period")

    @property
    def velocity(self):
        return np.r_[np.full(6, self.joint_speed_rad_s), self.gripper_speed_m_s]


@dataclass(frozen=True)
class NativeStartupRecovery:
    """Explicit firmware interpolation phase, separate from command slew limits."""
    verified_max_joint_speed_rad_s: tuple[float, ...]
    speed_percent: int = 1
    inset_rad: float = math.radians(1)
    max_travel_rad: float = math.radians(4)
    measured_speed_limit_rad_s: float = math.radians(5)
    nonrecover_drift_rad: float = math.radians(1)
    timeout_s: float = 20.0
    stable_s: float = .5
    tolerance_rad: float = math.radians(.25)
    check_inward_progress: bool = True

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, Mapping):
            raise ValueError("native_startup_recovery must be an object")
        unknown = set(value) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown native_startup_recovery fields: {sorted(unknown)}")
        if "verified_max_joint_speed_rad_s" not in value:
            raise ValueError("Native recovery requires verified_max_joint_speed_rad_s")
        speeds = _vector(value["verified_max_joint_speed_rad_s"], "verified firmware joint speed", 6)
        if np.any(speeds <= 0):
            raise ValueError("Verified firmware maximum joint speeds must be positive")
        speed = value.get("speed_percent", 1)
        if type(speed) is not int or speed != 1:
            raise ValueError("Native startup recovery requires speed_percent=1")
        check_inward_progress = value.get("check_inward_progress", True)
        if type(check_inward_progress) is not bool:
            raise ValueError("check_inward_progress must be an explicit boolean")
        kwargs = {field.name: _positive(value.get(field.name, field.default), f"native recovery {field.name}")
                  for field in fields(cls) if field.name not in (
                      "speed_percent", "verified_max_joint_speed_rad_s", "check_inward_progress")}
        upper_bounds = {"inset_rad": math.radians(1), "max_travel_rad": math.radians(4),
                        "measured_speed_limit_rad_s": math.radians(5),
                        "nonrecover_drift_rad": math.radians(1), "timeout_s": 20,
                        "tolerance_rad": math.radians(.25)}
        for name, upper in upper_bounds.items():
            if kwargs[name] > upper:
                raise ValueError(f"Native recovery {name} exceeds its supported bound")
        if not .5 <= kwargs["stable_s"] < kwargs["timeout_s"]:
            raise ValueError("Native recovery must satisfy .5 <= stable_s < timeout_s")
        if not kwargs["tolerance_rad"] < kwargs["inset_rad"] < kwargs["max_travel_rad"]:
            raise ValueError("Native recovery must satisfy tolerance_rad < inset_rad < max_travel_rad")
        if np.any(speeds * .01 >= kwargs["measured_speed_limit_rad_s"]):
            raise ValueError("Native 1% firmware speed must be below the measured speed guard")
        return cls(tuple(speeds), speed_percent=speed,
                   check_inward_progress=check_inward_progress, **kwargs)

    def as_dict(self):
        return {field.name: (list(self.verified_max_joint_speed_rad_s)
                            if field.name == "verified_max_joint_speed_rad_s" else getattr(self, field.name))
                for field in fields(self)}


@dataclass(frozen=True)
class ReplayPlan:
    actions: np.ndarray
    fps: float
    expected_initial_state: np.ndarray
    expected_initial_tolerance: np.ndarray
    source: dict
    robot_config: dict
    options: ReplayOptions
    native_startup_recovery: NativeStartupRecovery | None = None
    grasp_policy: GraspPolicy | None = None
    action_modes: tuple[str, ...] | None = None

    @property
    def command_bounds(self):
        return (np.r_[self.robot_config.get("joint_lower_rad", np.deg2rad([-154, 0, -175, -102, -75, -170])),
                      self.robot_config.get("gripper_min_width_m", 0.0)],
                np.r_[self.robot_config.get("joint_upper_rad", np.deg2rad([154, 195, 0, 102, 75, 170])),
                      self.robot_config.get("gripper_max_width_m", 0.07)])

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]):
        if data.get("schema_version") != 1:
            raise ValueError("Replay plan requires schema_version=1")
        raw = np.asarray(data["actions"])
        if raw.ndim != 2 or raw.shape[1] != 7 or not len(raw) or raw.dtype.kind not in "iuf":
            raise ValueError("actions must be a nonempty N x 7 absolute SI array")
        actions = raw.astype(np.float64)
        if not np.isfinite(actions).all():
            raise ValueError("actions must be finite")
        initial = _vector(data["expected_initial_state"], "expected_initial_state")
        tolerance = _vector(data["expected_initial_tolerance"], "expected_initial_tolerance")
        if np.any(tolerance <= 0) or np.any(tolerance > np.r_[np.full(6, math.radians(1)), 0.001] + 1e-12):
            raise ValueError("Initial tolerance must be positive and at most 1 degree / 1 mm")
        options = ReplayOptions(**dict(data.get("replay", {})))
        reference = options.execution_mode == "reference"
        if reference and "native_startup_recovery" in data:
            raise ValueError("Reference dispatch does not support native_startup_recovery")
        robot = dict(data["robot"])
        if robot.get("actuation_enabled") is not True:
            raise ValueError("Replay plan must explicitly set robot.actuation_enabled=true")
        lower = np.r_[_vector(robot.get("joint_lower_rad", np.deg2rad([-154, 0, -175, -102, -75, -170])), "joint_lower_rad", 6),
                      robot.get("gripper_min_width_m", 0.0)]
        upper = np.r_[_vector(robot.get("joint_upper_rad", np.deg2rad([154, 195, 0, 102, 75, 170])), "joint_upper_rad", 6),
                      robot.get("gripper_max_width_m", 0.07)]
        lower, upper = _vector(lower, "lower"), _vector(upper, "upper")
        if np.any(lower >= upper) or (not reference and (
                np.any(actions < lower - 1e-6) or np.any(actions > upper + 1e-6))):
            raise ValueError("Dataset actions exceed configured joint/gripper bounds")
        if reference:
            reference_limits(lower, upper)
        recovery = _vector(robot.get("joint_startup_recovery_tolerance_rad", [0.0] * 6), "startup recovery", 6)
        if np.any(recovery < 0) or np.any(recovery > math.radians(3) + 1e-12):
            raise ValueError("Startup recovery must be between 0 and 3 degrees")
        if reference:
            # Original client validates targets, not the initial joint reading.
            # Expected pose matching is still performed immediately before
            # enabling; accepting feedback never expands command bounds.
            if not lower[6] - 1e-7 <= initial[6] <= upper[6] + 1e-7:
                raise ValueError("Expected initial gripper state exceeds configured bounds")
            if robot.get("command_policy", "reference") != "reference":
                raise ValueError("Reference replay requires robot.command_policy=reference")
            robot["command_policy"] = "reference"
        else:
            if robot.get("command_policy", "projected") != "projected":
                raise ValueError("Tracked replay requires robot.command_policy=projected")
            if np.any(initial < lower - np.r_[recovery, 0] - 1e-7) or np.any(initial > upper + np.r_[recovery, 0] + 1e-7):
                raise ValueError("Expected initial state exceeds startup feedback bounds")
        hardware_velocity = np.r_[_vector(robot.get("max_joint_velocity_rad_s", [0.5] * 6), "max_joint_velocity_rad_s", 6),
                                  _positive(robot.get("max_gripper_velocity_m_s", .02), "hardware gripper velocity")]
        if np.any(options.velocity > hardware_velocity + 1e-12) or np.any(hardware_velocity <= 0):
            raise ValueError("Replay speed exceeds configured hardware command velocity")
        # A clipped startup-recovery target must not let the driver use a larger
        # speed cap than the reviewed replay speed. Narrowing never increases a
        # supplied hardware limit; the effective config is persisted in summary.
        robot["max_joint_velocity_rad_s"] = options.velocity[:6].tolist()
        robot["max_gripper_velocity_m_s"] = float(options.velocity[6])
        if abs(float(robot.get("control_hz", 30)) - options.control_hz) > 1e-9:
            raise ValueError("Replay and hardware control_hz must match")
        source = dict(data["source"])
        if not source:
            raise ValueError("Replay source identity is required")
        native = None
        if "native_startup_recovery" in data:
            native = NativeStartupRecovery.from_mapping(data["native_startup_recovery"])
            outside = (initial < lower - 1e-7) | (initial > upper + 1e-7)
            if np.any(outside[[0, 3, 4, 5, 6]]):
                raise ValueError("Native recovery supports only joints 2 and 3 outside nominal limits")
            # The hardware's outside-limit startup grace must cover this
            # explicitly bounded recovery; the global replay deadline remains.
            robot["startup_recovery_timeout_s"] = max(20.0, native.timeout_s,
                _positive(robot.get("startup_recovery_timeout_s", 10), "startup_recovery_timeout_s"))
        grasp_policy = action_modes = None
        if ("grasp_policy" in data) != ("action_modes" in data):
            raise ValueError("grasp_policy and action_modes must be provided together")
        if "grasp_policy" in data:
            grasp_policy = GraspPolicy.from_mapping(data["grasp_policy"], lower[6], upper[6])
            if reference:
                modes = data["action_modes"]
                if (not isinstance(modes, (list, tuple)) or len(modes) != len(actions)
                        or any(type(mode) is not str or mode not in ("position", "close", "carry", "release")
                               for mode in modes)):
                    raise ValueError("action_modes must contain one known mode per action")
                action_modes = tuple(modes)
            else:
                action_modes = validate_grasp_program(actions, data["action_modes"], grasp_policy)
            if not reference and grasp_policy.arm_tracking_tolerance_rad > options.max_tracking_error_rad:
                raise ValueError("Grasp arm tracking tolerance must remain below the hard bound")
            if not reference and options.max_tracking_error_gripper_m > .015:
                raise ValueError("Grasp-aware replay retains a gripper hard bound of at most 15 mm")
        return cls(actions, _positive(data["fps"], "fps"), initial, tolerance, source, robot,
                   options, native, grasp_policy, action_modes)

    def summary(self):
        reference = self.options.execution_mode == "reference"
        if reference:
            chunk_size = self.options.reference_chunk_size or len(self.actions)
            chunk_count = math.ceil(len(self.actions) / chunk_size)
            duration = len(self.actions) / self.options.control_hz + chunk_count * REFERENCE_SETTLE_TIME_S
        else:
            previous = np.vstack((self.expected_initial_state, self.actions[:-1]))
            minimum = np.r_[1.0 / self.options.control_hz,
                            np.full(len(self.actions) - 1, 1.0 / self.fps)]
            durations = np.maximum(minimum, np.max(np.abs(self.actions - previous) / self.options.velocity, axis=1))
            duration = float(durations.sum())
        result = {"validated": True, "action_count": len(self.actions), "source": self.source,
                "source_duration_s": (len(self.actions) - 1) / self.fps,
                "minimum_replay_duration_s": duration,
                "max_wall_time_s": self.options.max_wall_time_s, "robot": self.robot_config,
                "units": "arm radians; full gripper width meters",
                "execution_mode": self.options.execution_mode}
        if reference:
            lower, upper = reference_limits(*self.command_bounds)
            result.update({
                "success_meaning": "dispatch_only", "dispatch_complete": False,
                "target_reached": None, "task_success": None,
                "reference_source": dict(REFERENCE_SOURCE),
                "reference_dispatch": {"control_hz": self.options.control_hz,
                    "chunk_size": chunk_size, "chunk_count": chunk_count,
                    "settle_time_s": REFERENCE_SETTLE_TIME_S,
                    "max_delta_si": REFERENCE_MAX_DELTA.tolist(),
                    "target_lower_si": lower.tolist(), "target_upper_si": upper.tolist(),
                    "arrival_wait": False, "contact_gating": False,
                    "automatic_smoothing": False, "grasp_metadata_used_for_dispatch": False,
                    "adapter_policy": "reference",
                    "interpolation": "firmware MOVEJ at robot.speed_percent; no adapter velocity slew guarantee",
                    "measured_joint_bounds_enforced": False,
                    "cleanup": "no new target; firmware retains last applied goal; cancellation/fault still stops"},
                "nominal_dispatch_duration_s": duration,
                "duration_note": "Action count / control_hz + chunk count * fixed settling time; "
                    "this is a nominal queue schedule, not an arrival estimate. Driver latency may lengthen it.",
                "completion_note": "Dispatch completion does not establish measured target arrival, grasp, "
                    "placement, or task success. Target bounds and fault/freshness handling remain; "
                    "joint_speed_rad_s and gripper_speed_m_s are not reference-mode physical speed guarantees."})
        if self.native_startup_recovery is not None:
            result["native_startup_recovery"] = {
                **self.native_startup_recovery.as_dict(),
                "control_mode": "firmware-native interpolation; not normal command slew limiting",
                "target_rule": "Derived from fresh measured state; only recovering joints move inward, other joints and gripper hold.",
                "feedback_policy": ("monotonic inward progress with shrinking feedback slack"
                                    if self.native_startup_recovery.check_inward_progress else
                                    "fixed configured startup tolerance for original J2/J3; native phase only"),
                "duration_note": "minimum_replay_duration_s excludes native recovery and estimates approach from expected initial state; actual approach starts from fresh recovered feedback.",
            }
        if self.grasp_policy is not None:
            result["grasp_policy"] = self.grasp_policy.as_dict()
            result["action_modes"] = list(self.action_modes)
            result["grasp_evidence_note"] = ("Recorded only; reference dispatch does not run GraspMonitor "
                "or use contact events to gate this trajectory." if reference else
                "Loaded contact within an explicit width interval; this is not object identity or placement verification. "
                "Acquired close/carry phases and a bounded release opening bridge below verified contact width "
                "may exempt soft gripper arrival; all arm checks and hard bounds remain.")
        return result


class ReplayWatchdog:
    """Independent progress/cancellation monitor; stop transmission may block."""

    def __init__(self, robot, timeout_s, deadline, cancel, stop_file, monotonic=time.monotonic):
        self.robot, self.timeout_s, self.deadline = robot, timeout_s, deadline
        self.cancel, self.stop_file, self.monotonic = cancel, stop_file, monotonic
        self.heartbeat = monotonic()
        self.failure = None
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, name="real-replay-watchdog", daemon=True)

    def check(self):
        now = self.monotonic()
        if self.cancel.is_set() or (self.stop_file is not None and self.stop_file.exists()):
            return "Replay cancelled"
        if now >= self.deadline:
            return "Replay wall-time limit exceeded"
        if now - self.heartbeat > self.timeout_s:
            return "Replay control heartbeat timed out"
        return None

    def _run(self):
        while not self.done.wait(min(.025, self.timeout_s / 4)):
            failure = self.check()
            if failure is not None:
                self.failure = failure
                # The adapter also checks this event immediately before enable,
                # including when a stop races with connect resetting its latch.
                self.cancel.set()
                try:
                    self.robot.stop()
                except BaseException as error:
                    self.failure += f"; stop failed: {error}"
                return

    def start(self):
        self.thread.start()

    def close(self):
        self.done.set()
        self.thread.join(timeout=1)


def _dispatch_reference(plan, robot, watchdog, log, report, *, started,
                        monotonic, sleep, wall_time):
    """Execute the original client queue rules, without arrival/contact gates."""
    options = plan.options
    period = 1.0 / options.control_hz
    chunk_size = options.reference_chunk_size or len(plan.actions)
    lower, upper = reference_limits(*plan.command_bounds)
    next_tick = monotonic()

    def check():
        error = watchdog.failure or watchdog.check()
        if error:
            raise ReplayError(error)

    def tick_start():
        wait = next_tick - monotonic()
        if wait > 0:
            sleep(wait)
        check()
        return monotonic()

    def send(target, observed, *, index, chunk_index, phase, tick, settle_elapsed=None):
        nonlocal next_tick
        applied = _vector(robot.command(target), "applied state")
        check()
        record = {"unix_s": wall_time(), "elapsed_s": monotonic() - started,
                  "phase": phase, "execution_mode": "reference",
                  "action_index": index, "chunk_index": chunk_index,
                  "target": target.tolist(), "applied": applied.tolist(),
                  "observed": None if observed is None else observed.tolist(),
                  "observation_timing": "before_command" if observed is not None else None,
                  "tracking_wait": False, "arrival_checked": False,
                  "settle_elapsed_s": settle_elapsed}
        log.write(json.dumps(record, allow_nan=False) + "\n")
        log.flush()
        watchdog.heartbeat = monotonic()
        report["ticks"] += 1
        # Original loop sleeps only for the unspent part of this period.
        # Slow calls lengthen the schedule; no missed ticks are replayed.
        next_tick = tick + period

    for chunk_index, begin in enumerate(range(0, len(plan.actions), chunk_size)):
        end = min(begin + chunk_size, len(plan.actions))
        for index in range(begin, end):
            tick = tick_start()
            measured = _vector(robot.read_state(), "reference measured state")
            check()
            target = reference_target(plan.actions[index], measured, lower, upper)
            send(target, measured, index=index, chunk_index=chunk_index,
                 phase="reference_dispatch", tick=tick)
            report["completed_actions"] = index + 1
        # Repeat the last already clipped/delta-limited target, rather than
        # applying a new delta against each settling observation. The original
        # client also sends that target on the tick where settling first ends;
        # the next chunk begins on the following tick.
        settle_started = monotonic()
        while True:
            tick = tick_start()
            elapsed = tick - settle_started
            send(target, None, index=end - 1, chunk_index=chunk_index,
                 phase="reference_settling", tick=tick, settle_elapsed=elapsed)
            if elapsed >= REFERENCE_SETTLE_TIME_S:
                break
    report["dispatch_complete"] = True


def execute_replay(plan: ReplayPlan | Mapping[str, Any], robot_factory: Callable, output_dir,
                   *, stop_file=None, cancel=None, monotonic=time.monotonic, sleep=time.sleep,
                   wall_time=time.time):
    """Execute a prevalidated plan; persist failure evidence and always clean up.

    The caller owns the explicit actuation permission. Validation, existing-stop
    checks, and output creation occur before construction/connection. On success
    the tracked robot holds its measured final pose; reference mode leaves the
    existing firmware target unchanged. Motors are never automatically
    disabled. Expected grasp failures hold the arm, retaining the applied clamp
    target when contact was acquired. Hardware/control failures and cancellation
    latch ``stop`` before disconnection.
    """
    if not isinstance(plan, ReplayPlan):
        plan = ReplayPlan.from_mapping(plan)
    else:
        # Frozen dataclasses can still contain mutable arrays/dicts. Revalidate
        # at the actuation boundary even when the caller inspected the plan.
        if (plan.grasp_policy is None) != (plan.action_modes is None):
            raise ValueError("grasp_policy and action_modes must both be absent or both be present")
        plan_data = {"schema_version": 1, "actions": plan.actions,
            "fps": plan.fps, "expected_initial_state": plan.expected_initial_state,
            "expected_initial_tolerance": plan.expected_initial_tolerance,
            "source": plan.source, "robot": plan.robot_config,
            "replay": {field.name: getattr(plan.options, field.name) for field in fields(plan.options)}}
        if plan.native_startup_recovery is not None:
            plan_data["native_startup_recovery"] = plan.native_startup_recovery.as_dict()
        if plan.grasp_policy is not None:
            plan_data["grasp_policy"] = plan.grasp_policy.as_dict()
            plan_data["action_modes"] = plan.action_modes
        plan = ReplayPlan.from_mapping(plan_data)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path, log_path = output / "summary.json", output / "replay.jsonl"
    if summary_path.exists() or log_path.exists():
        raise FileExistsError("Refusing to overwrite replay evidence")
    cancel = cancel if cancel is not None else threading.Event()
    stop_file = Path(stop_file) if stop_file is not None else None
    if cancel.is_set() or (stop_file is not None and stop_file.exists()):
        raise ReplayError("Replay was cancelled before connection")
    options = plan.options
    reference = options.execution_mode == "reference"
    robot = robot_factory(plan.robot_config)
    started = monotonic()
    deadline = started + options.max_wall_time_s
    report = {**plan.summary(), "success": False, "started_unix_s": wall_time(),
              "completed_actions": 0, "ticks": 0, "error": None, "cleanup_errors": []}
    watchdog = None
    failure = None
    grasp = (None if reference or plan.grasp_policy is None else GraspMonitor(
        plan.grasp_policy, plan.action_modes, plan.command_bounds[0][6], plan.command_bounds[1][6]))
    with log_path.open("x") as log:
        try:
            if (plan.native_startup_recovery is not None
                    and not callable(getattr(robot, "recover_startup_native", None))):
                raise ReplayError("Robot adapter does not support native startup recovery")
            if grasp is not None and not callable(getattr(robot, "read_gripper_feedback", None)):
                raise ReplayError("Grasp-aware replay requires read_gripper_feedback support")
            if grasp is not None and not callable(getattr(robot, "hold_grasp", None)):
                raise ReplayError("Grasp-aware replay requires hold_grasp support for preserving the clamp goal")
            startup_budget = (_positive(plan.robot_config.get("connect_timeout_s", 3), "connect_timeout_s")
                              + _positive(plan.robot_config.get("enable_timeout_s", 3), "enable_timeout_s") + .5)
            watchdog = ReplayWatchdog(robot, startup_budget, deadline, cancel, stop_file, monotonic)
            watchdog.start()
            robot.connect(expected_state=plan.expected_initial_state,
                          expected_tolerance=plan.expected_initial_tolerance,
                          cancel_event=cancel)
            if watchdog.failure or watchdog.check():
                raise ReplayError(watchdog.failure or watchdog.check())
            watchdog.heartbeat = monotonic()
            watchdog.timeout_s = options.watchdog_timeout_s
            observed = _vector(robot.read_state(), "observed state")
            report["initial_state"] = observed.tolist()
            if plan.native_startup_recovery is not None:
                native = plan.native_startup_recovery
                recover = robot.recover_startup_native
                report["native_recovery_completed"] = False
                report["native_recovery_feedback_samples"] = 0
                native_started = monotonic()

                def native_progress(snapshot):
                    error = watchdog.failure or watchdog.check()
                    if error:
                        raise ReplayError(error)
                    if monotonic() - native_started >= native.timeout_s:
                        raise ReplayError("Native startup recovery time limit exceeded")
                    if not isinstance(snapshot, Mapping):
                        raise ReplayError("Native recovery callback must provide a feedback snapshot")
                    feedback = _vector(snapshot["state"], "native recovery feedback")
                    native_target = _vector(snapshot["target"], "native recovery target")
                    lower, upper = plan.command_bounds
                    if np.any(native_target < lower - 1e-7) or np.any(native_target > upper + 1e-7):
                        raise ReplayError("Native recovery generated target outside nominal limits")
                    record = {"unix_s": wall_time(), "elapsed_s": monotonic() - started,
                              "phase": "native_recovery", "target": native_target.tolist(),
                              "observed": feedback.tolist(), "speed_percent": native.speed_percent,
                              "check_inward_progress": native.check_inward_progress,
                              "control_mode": "firmware-native interpolation",
                              "native_elapsed_s": snapshot.get("elapsed_s"),
                              "stable_elapsed_s": snapshot.get("stable_elapsed_s"),
                              "recovery_joint_indices": snapshot.get("recovery_joint_indices"),
                              "feedback_timestamps": snapshot.get("feedback_timestamps")}
                    measured_speed = snapshot.get("measured_speed_rad_s")
                    record["measured_speed_rad_s"] = (None if measured_speed is None else
                        _vector(measured_speed, "native recovery measured speed", 6).tolist())
                    log.write(json.dumps(record, allow_nan=False) + "\n")
                    log.flush()
                    watchdog.heartbeat = monotonic()
                    report["native_recovery_feedback_samples"] += 1

                native_result = recover(**native.as_dict(), progress_callback=native_progress, cancel_event=cancel)
                if not isinstance(native_result, Mapping) or native_result.get("completed") is not True:
                    raise ReplayError("Native startup recovery did not confirm completion")
                # Reject malformed evidence before entering the normal command
                # loop or allowing the final summary serializer to fail later.
                json.dumps(native_result, allow_nan=False)
                error = watchdog.failure or watchdog.check()
                if error:
                    raise ReplayError(error)
                if monotonic() - native_started >= native.timeout_s:
                    raise ReplayError("Native startup recovery time limit exceeded")
                # Native motion is finished before normal interpolation starts;
                # use actual fresh feedback, never the planned recovery target.
                observed = _vector(robot.read_state(), "post-recovery observed state")
                error = watchdog.failure or watchdog.check()
                if error:
                    raise ReplayError(error)
                lower, upper = plan.command_bounds
                if np.any(observed < lower - 1e-7) or np.any(observed > upper + 1e-7):
                    raise ReplayError("Native startup recovery returned outside nominal joint limits")
                report["native_recovery_completed"] = True
                report["native_recovery_elapsed_s"] = monotonic() - native_started
                report["post_native_recovery_state"] = observed.tolist()
                report["native_recovery_result"] = native_result
                watchdog.heartbeat = monotonic()
            if reference:
                _dispatch_reference(plan, robot, watchdog, log, report, started=started,
                                    monotonic=monotonic, sleep=sleep, wall_time=wall_time)
            tight = np.r_[np.full(6, options.tracking_tolerance_rad), options.tracking_tolerance_gripper_m]
            hard = np.r_[np.full(6, options.max_tracking_error_rad), options.max_tracking_error_gripper_m]
            period = 1.0 / options.control_hz
            previous = observed.copy()
            last_applied = observed.copy()
            next_tick = monotonic()
            for index, destination in enumerate(() if reference else plan.actions):
                if grasp is not None:
                    grasp.begin_action(index, monotonic())
                duration = max(period if index == 0 else 1.0 / plan.fps,
                               float(np.max(np.abs(destination - previous) / options.velocity)))
                steps = max(1, math.ceil(duration / period))
                interpolation_step = 1
                lag_started = None
                while True:
                    wait = next_tick - monotonic()
                    if wait > 0:
                        sleep(wait)
                    error = watchdog.failure or watchdog.check()
                    if error:
                        raise ReplayError(error)
                    tick_start = monotonic()
                    close_arm_wait = False
                    if grasp is not None:
                        # Inspect carry loss before sending another motion. The
                        # loaded-contact test uses the last actually applied
                        # clamp goal, not a future or interpolated destination.
                        observed = _vector(robot.read_state(), "observed state")
                        if np.any(np.abs(observed - last_applied) > hard):
                            raise ReplayError("Measured tracking error exceeds hard joint/gripper bound")
                        arm_ready = bool(np.all(np.abs(observed[:6] - destination[:6])
                                               <= min(grasp.policy.arm_tracking_tolerance_rad, tight[0])))
                        grasp.observe(robot.read_gripper_feedback(), arm_ready=arm_ready,
                                      applied_gripper=last_applied[6], now=monotonic(), wall_time=wall_time(),
                                      hard_gripper_error_m=hard[6])
                        grasp.check_release_bridge_deadline(monotonic(), options.tracking_timeout_s)
                        close_arm_wait = grasp.mode == "close" and not arm_ready
                        error = watchdog.failure or watchdog.check()
                        if error:
                            raise ReplayError(error)
                    interpolated = previous + (destination - previous) * (interpolation_step / steps)
                    # Startup feedback can be slightly outside nominal limits.
                    # Request an in-range target; the driver alone owns bounded
                    # inward recovery from the measured pose. Dataset actions
                    # themselves were validated and are never clipped here.
                    target = np.clip(interpolated, *plan.command_bounds) if index == 0 else interpolated
                    if grasp is not None and grasp.mode == "close":
                        target[:6] = destination[:6]
                        if close_arm_wait:
                            # Maintain the last clamp/opening goal while the
                            # stationary close pose settles; do not close early.
                            target[6] = last_applied[6]
                    applied = _vector(robot.command(target), "applied state")
                    last_applied = applied.copy()
                    observed = _vector(robot.read_state(), "observed state")
                    now = monotonic()
                    if watchdog.failure or watchdog.check():
                        raise ReplayError(watchdog.failure or watchdog.check())
                    if np.any(np.abs(observed - applied) > hard):
                        raise ReplayError("Measured tracking error exceeds hard joint/gripper bound")
                    if grasp is None:
                        lagging = timeout_lag = bool(np.any(np.abs(observed - target) > tight))
                    else:
                        arm_tolerance = (min(grasp.policy.arm_tracking_tolerance_rad, tight[0])
                                         if grasp.mode == "close" else tight[0])
                        arm_lag = bool(np.any(np.abs(observed[:6] - target[:6]) > arm_tolerance))
                        release_exempt = grasp.release_soft_gripper_exempt(target[6], observed[6])
                        grip_lag = bool(abs(observed[6] - target[6]) > tight[6]
                                        and not grasp.soft_gripper_exempt and not release_exempt)
                        phase_wait = interpolation_step == steps and grasp.phase_wait()
                        lagging = arm_lag or grip_lag or phase_wait or close_arm_wait
                        # Expected contact gets its own finite close deadline.
                        # Arm tracking and every other mode retain the existing
                        # timeout, including normal gripper arrival on release.
                        timeout_lag = arm_lag or close_arm_wait or (
                            grasp.mode != "close" and (grip_lag or phase_wait))
                    if timeout_lag:
                        lag_started = now if lag_started is None else lag_started
                        if now - lag_started >= options.tracking_timeout_s:
                            raise ReplayError("Robot did not catch up within tracking timeout")
                    else:
                        lag_started = None
                    record = {"unix_s": wall_time(), "elapsed_s": now - started,
                              "phase": "approach" if index == 0 else "dataset", "action_index": index,
                              "fraction": interpolation_step / steps, "target": target.tolist(),
                              "interpolated_target": interpolated.tolist(),
                              "applied": applied.tolist(), "observed": observed.tolist(),
                              "tracking_wait": lagging}
                    if grasp is not None:
                        record["action_mode"] = grasp.mode
                        record["grasp"] = grasp.snapshot()
                    log.write(json.dumps(record, allow_nan=False) + "\n")
                    log.flush()
                    watchdog.heartbeat = monotonic()
                    report["ticks"] += 1
                    # Never burst commands to catch up after a slow driver call.
                    next_tick = max(tick_start + period, monotonic() + period)
                    if not lagging:
                        if interpolation_step == steps:
                            break
                        interpolation_step += 1
                previous = destination.copy()
                if grasp is not None:
                    grasp.finish_action()
                report["completed_actions"] = index + 1
            robot.hold()
            if watchdog.failure or watchdog.check():
                raise ReplayError(watchdog.failure or watchdog.check())
            report["final_state"] = _vector(robot.read_state(), "final state").tolist()
            if watchdog.failure or watchdog.check():
                raise ReplayError(watchdog.failure or watchdog.check())
            report["success"] = True
        except GraspTaskFailure as error:
            failure = error
            report["error"] = f"{type(error).__name__}: {error}"
            report["failure_kind"] = "task"
            report["task_outcome"] = error.outcome
            report["emergency_stop_requested"] = False
            log.write(json.dumps({"unix_s": wall_time(), "elapsed_s": monotonic() - started,
                                  "phase": "task_failure", "outcome": error.outcome,
                                  "grasp": grasp.snapshot()}, allow_nan=False) + "\n")
            log.flush()
            try:
                # Empty/wrong-width contact or lost load is a task failure.
                # Holding avoids intentionally invoking the firmware emergency
                # mode, which can allow the arm to descend under gravity.
                if grasp.acquired and grasp.mode in ("close", "carry"):
                    # This adapter operation also preserves the exact clamp goal
                    # during close(); a measured-width hold would unload contact.
                    robot.hold_grasp(gripper_target_m=float(last_applied[6]))
                else:
                    robot.hold()
                if watchdog.failure or watchdog.check():
                    raise ReplayError(watchdog.failure or watchdog.check())
            except BaseException as hold_error:
                report["failure_kind"] = "hardware_or_control"
                report["cleanup_errors"].append(f"task-failure hold: {hold_error}")
                report["emergency_stop_requested"] = True
                try:
                    robot.stop()
                except BaseException as stop_error:
                    report["cleanup_errors"].append(f"stop: {stop_error}")
        except BaseException as error:
            failure = error
            report["error"] = f"{type(error).__name__}: {error}"
            if grasp is not None or reference:
                report["failure_kind"] = "hardware_or_control"
                report["emergency_stop_requested"] = True
            try:
                robot.stop()
            except BaseException as stop_error:
                report["cleanup_errors"].append(f"stop: {stop_error}")
        finally:
            if watchdog is not None:
                # SDK disconnect joins receive threads; keep a finite, separate
                # cleanup budget while monitoring cancellation and global time.
                watchdog.timeout_s = options.close_timeout_s
                watchdog.heartbeat = monotonic()
            try:
                robot.close()
            except BaseException as close_error:
                report["cleanup_errors"].append(f"close: {close_error}")
                report["success"] = False
                if grasp is not None or reference:
                    report["failure_kind"] = "hardware_or_control"
                    report["emergency_stop_requested"] = True
                try:
                    robot.stop()
                except BaseException as stop_error:
                    report["cleanup_errors"].append(f"stop after close failure: {stop_error}")
                if failure is None:
                    failure = close_error
                    report["error"] = f"Close failed: {close_error}"
            if watchdog is not None:
                if watchdog.failure or watchdog.check():
                    cleanup_fault = watchdog.failure or watchdog.check()
                    report["cleanup_errors"].append(f"watchdog: {cleanup_fault}")
                    report["success"] = False
                    if grasp is not None or reference:
                        report["failure_kind"] = "hardware_or_control"
                        report["emergency_stop_requested"] = True
                    if reference:
                        # Synchronous detection must issue the watchdog stop
                        # even if its worker has not run before thread cleanup.
                        try:
                            robot.stop()
                        except BaseException as stop_error:
                            report["cleanup_errors"].append(f"stop after cleanup watchdog failure: {stop_error}")
                    if failure is None:
                        failure = ReplayError(cleanup_fault)
                        report["error"] = cleanup_fault
                watchdog.close()
            report["elapsed_s"] = monotonic() - started
            report["finished_unix_s"] = wall_time()
            if grasp is not None:
                report["grasp"] = grasp.snapshot()
            summary_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if failure is not None:
        error_type = ReplayTaskFailure if report.get("failure_kind") == "task" else ReplayError
        raise error_type(f"Replay failed; see {summary_path}: {report['error']}") from failure
    return report
