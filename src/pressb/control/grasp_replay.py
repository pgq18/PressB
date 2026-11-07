"""Optional contact-aware gripper state machine for finite physical replay.

Contact is evidence of a loaded grasp within a reviewed width interval, not
object recognition. The normal robot fault, freshness and hard tracking guards
remain authoritative. This module never sends commands or adjusts motor effort.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from typing import Mapping

import numpy as np


class GraspTaskFailure(RuntimeError):
    """A bounded task failure requiring hold, not a firmware emergency stop."""
    def __init__(self, outcome, message):
        self.outcome = outcome
        super().__init__(message)


class GraspFeedbackError(RuntimeError):
    """Invalid hardware evidence; the caller must use its fault/stop path."""


def _number(value, name, *, positive=True):
    if isinstance(value, (bool, np.bool_, str)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be a {'positive ' if positive else ''}finite number")
    return result


@dataclass(frozen=True)
class GraspPolicy:
    expected_width_m: tuple[float, float]
    min_effort_raw: int
    contact_dwell_s: float = .3
    max_width_speed_m_s: float = .003
    carry_loss_grace_s: float = .15
    max_feedback_age_s: float = .15
    close_timeout_s: float = 5.0
    min_closing_gap_m: float = .001
    arm_stationary_tolerance_rad: float = math.radians(.1)
    arm_tracking_tolerance_rad: float = math.radians(.5)
    max_effort_raw: int = 32767

    @classmethod
    def from_mapping(cls, data, lower_width, upper_width):
        if not isinstance(data, Mapping):
            raise ValueError("grasp_policy must be an object")
        unknown = set(data) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown grasp_policy fields: {sorted(unknown)}")
        if "expected_width_m" not in data or "min_effort_raw" not in data:
            raise ValueError("grasp_policy requires expected_width_m and min_effort_raw")
        width = np.asarray(data["expected_width_m"])
        if width.shape != (2,) or width.dtype.kind not in "iuf" or not np.isfinite(width).all():
            raise ValueError("expected_width_m must contain two finite widths")
        width = tuple(float(value) for value in width)
        if not lower_width < width[0] < width[1] < upper_width:
            raise ValueError("Expected contact interval must be strictly inside gripper bounds")
        lo_effort, hi_effort = data["min_effort_raw"], data.get("max_effort_raw", 32767)
        if type(lo_effort) is not int or type(hi_effort) is not int or not 0 < lo_effort <= hi_effort <= 32767:
            raise ValueError("Effort thresholds must be integers with 0 < min <= max <= 32767")
        kwargs = {field.name: _number(data.get(field.name, field.default), field.name)
                  for field in fields(cls) if field.name not in ("expected_width_m", "min_effort_raw", "max_effort_raw")}
        limits = {"contact_dwell_s": (.2, 2), "max_width_speed_m_s": (.00001, .01),
                  "carry_loss_grace_s": (.01, .3), "max_feedback_age_s": (.02, .25),
                  "close_timeout_s": (.5, 30), "min_closing_gap_m": (.0001, .005),
                  "arm_stationary_tolerance_rad": (1e-6, math.radians(.5)),
                  "arm_tracking_tolerance_rad": (1e-6, math.radians(1))}
        for name, (lower, upper) in limits.items():
            if not lower <= kwargs[name] <= upper:
                raise ValueError(f"{name} must be in [{lower}, {upper}]")
        if kwargs["close_timeout_s"] <= kwargs["contact_dwell_s"]:
            raise ValueError("close_timeout_s must exceed contact_dwell_s")
        if width[1] + kwargs["min_closing_gap_m"] > upper_width:
            raise ValueError("Expected width leaves insufficient opening clearance for release")
        return cls(width, lo_effort, max_effort_raw=hi_effort, **kwargs)

    def as_dict(self):
        result = asdict(self)
        result["expected_width_m"] = list(self.expected_width_m)
        return result


def validate_grasp_program(actions, modes, policy):
    if not isinstance(modes, (list, tuple)) or len(modes) != len(actions):
        raise ValueError("action_modes must match the number of actions")
    if any(type(mode) is not str or mode not in ("position", "close", "carry", "release") for mode in modes):
        raise ValueError("action_modes must contain position, close, carry or release")
    if modes[0] != "position":
        raise ValueError("Grasp-aware replay must first position the arm")
    phase, clamp_goal, close_pose, release_pose = "position", None, None, None
    acquisitions = 0
    for index, mode in enumerate(modes):
        previous_mode = modes[index - 1] if index else None
        if mode == "position":
            if phase in ("close", "carry"):
                raise ValueError("A grasp must be released before returning to position mode")
            phase = "position"
        elif mode == "close":
            if previous_mode not in ("position", "close"):
                raise ValueError("close must follow position or another close action")
            if previous_mode == "position":
                acquisitions += 1
                close_pose = actions[index - 1, :6]
            if np.any(np.abs(actions[index, :6] - close_pose) > policy.arm_stationary_tolerance_rad):
                raise ValueError("Arm targets must remain stationary throughout close")
            if actions[index, 6] > actions[index - 1, 6] + 1e-6:
                raise ValueError("close gripper targets must not open")
            phase, clamp_goal = "close", float(actions[index, 6])
            if (index + 1 == len(modes) or modes[index + 1] != "close"):
                if clamp_goal > policy.expected_width_m[0] - policy.min_closing_gap_m + 1e-9:
                    raise ValueError("Final close goal must apply closing pressure below the expected contact interval")
        elif mode == "carry":
            if phase not in ("close", "carry"):
                raise ValueError("carry requires a preceding close phase")
            if abs(actions[index, 6] - clamp_goal) > 1e-6:
                raise ValueError("carry must preserve the final close clamp goal")
            phase = "carry"
        else:
            if phase not in ("close", "carry", "release"):
                raise ValueError("release requires a preceding grasp")
            if previous_mode != "release":
                release_pose = actions[index - 1, :6]
            if np.any(np.abs(actions[index, :6] - release_pose) > policy.arm_stationary_tolerance_rad):
                raise ValueError("Arm targets must remain stationary throughout release; reposition after release")
            if actions[index, 6] < actions[index - 1, 6] - 1e-6:
                raise ValueError("release gripper targets must not close")
            phase = "release"
            if (index + 1 == len(modes) or modes[index + 1] != "release"):
                if actions[index, 6] < policy.expected_width_m[1] + policy.min_closing_gap_m - 1e-9:
                    raise ValueError("Final release goal must open beyond the expected contact interval")
    if phase in ("close", "carry") or acquisitions == 0:
        raise ValueError("A finite grasp program requires close and release; it cannot disconnect while carrying")
    return tuple(modes)


class GraspMonitor:
    """Observe immutable gripper frames and gate motion from explicit phases."""
    def __init__(self, policy, modes, lower_width, upper_width):
        self.policy, self.modes = policy, modes
        self.lower_width, self.upper_width = lower_width, upper_width
        self.mode = "position"
        self.index = -1
        self.acquired = False
        self.acquisitions = self.releases = 0
        self.close_started = None
        self.close_active_s = 0.0
        self._last_close_observation = None
        self._close_was_ready = False
        self.candidate_started = None
        self.loss_started = None
        self.previous_feedback = None
        self.feedback = None
        self.width_speed = None
        self.contact_valid = False
        self.outcome = "not_started"
        self.last_contact_width_m = None
        self.release_started = None
        self.release_contact_width_m = None
        self.release_bridge_active = False
        self.release_bridge_exempt = False

    def begin_action(self, index, now):
        new_mode = self.modes[index]
        if new_mode == "close" and self.mode != "close":
            self.acquired = False
            self.close_started, self.candidate_started, self.loss_started = None, None, None
            self.close_active_s = 0.0
            self._last_close_observation, self._close_was_ready = now, False
            self.last_contact_width_m = None
            self.outcome = "awaiting_contact"
        if new_mode == "carry" and not self.acquired:
            raise GraspTaskFailure("grasp_not_acquired", "Carry blocked: no acquired grasp")
        if new_mode == "release" and self.mode != "release":
            if not self.acquired:
                raise GraspTaskFailure("grasp_not_acquired", "Release reached without an acquired grasp")
            # Keep only a previously verified loaded width, never a possibly
            # invalid sample seen during the carry-loss grace period. Opening
            # initially still commands below this width and cannot reach that
            # positional target while the object remains between the fingers.
            self.release_started = now
            self.release_contact_width_m = self.last_contact_width_m
            self.release_bridge_active = self.release_contact_width_m is not None
            self.release_bridge_exempt = False
            self.acquired, self.contact_valid = False, False
            self.candidate_started = self.loss_started = None
            self.outcome = "releasing"
        self.mode, self.index = new_mode, index

    def _parse_feedback(self, value, wall_time):
        if not isinstance(value, Mapping):
            raise GraspFeedbackError("Gripper feedback must be an object")
        required = {"width_m", "effort_raw", "status_raw", "enabled", "fault", "timestamp_s"}
        if not required.issubset(value):
            raise GraspFeedbackError("Gripper feedback is missing required fields")
        try:
            width = _number(value["width_m"], "gripper width", positive=False)
            stamp = _number(value["timestamp_s"], "gripper timestamp", positive=False)
            if not self.lower_width - 1e-7 <= width <= self.upper_width + 1e-7:
                raise ValueError("Gripper width is outside hardware limits")
            effort, status = value["effort_raw"], value["status_raw"]
            if type(effort) is not int or not -32768 <= effort <= 32767:
                raise ValueError("effort_raw must be signed int16")
            if type(status) is not int or not 0 <= status <= 255:
                raise ValueError("status_raw must be uint8")
            if type(value["enabled"]) is not bool or type(value["fault"]) is not bool:
                raise ValueError("Gripper enabled/fault fields must be booleans")
            if value["enabled"] != bool(status & 0x40) or value["fault"] != bool(status & 0x3f):
                raise ValueError("Gripper status fields disagree with status_raw")
        except ValueError as error:
            raise GraspFeedbackError(str(error)) from error
        if value["fault"] or not value["enabled"]:
            raise GraspFeedbackError("Gripper reports a fault or lost enable")
        if abs(effort) > self.policy.max_effort_raw:
            raise GraspFeedbackError("Gripper effort exceeds configured maximum")
        if wall_time - stamp > self.policy.max_feedback_age_s or stamp - wall_time > .05:
            raise GraspFeedbackError("Gripper feedback timestamp is stale or in the future")
        if self.previous_feedback is not None and stamp < self.previous_feedback["timestamp_s"]:
            raise GraspFeedbackError("Gripper feedback timestamp moved backwards")
        return {"width_m": width, "effort_raw": effort, "status_raw": status,
                "enabled": value["enabled"], "fault": value["fault"], "timestamp_s": stamp}

    def observe(self, feedback, *, arm_ready, applied_gripper, now, wall_time, hard_gripper_error_m=.015):
        current = self._parse_feedback(feedback, wall_time)
        if abs(current["width_m"] - applied_gripper) > hard_gripper_error_m:
            raise GraspFeedbackError("Gripper feedback exceeds hard joint/gripper bound")
        previous = self.previous_feedback
        new_frame = previous is None or current["timestamp_s"] > previous["timestamp_s"]
        if not new_frame and current != previous:
            raise GraspFeedbackError("Gripper values changed without a new frame timestamp")
        self.feedback = current
        if new_frame:
            self.width_speed = (None if previous is None else abs(current["width_m"] - previous["width_m"])
                                / (current["timestamp_s"] - previous["timestamp_s"]))
            self.previous_feedback = current
        lo, hi = self.policy.expected_width_m
        self.contact_valid = bool(lo <= current["width_m"] <= hi
                              and abs(current["effort_raw"]) >= self.policy.min_effort_raw
                              and current["width_m"] - applied_gripper >= self.policy.min_closing_gap_m)
        if self.mode == "close" and not self.acquired:
            # The close deadline measures time with the arm settled. Unsettled
            # arm feedback remains the runner's tracking-fault responsibility,
            # rather than being mislabeled as an ordinary empty grasp.
            if arm_ready:
                if self.close_started is None:
                    self.close_started = now
                if self._close_was_ready:
                    self.close_active_s += now - self._last_close_observation
            else:
                self.candidate_started = None
            self._last_close_observation, self._close_was_ready = now, arm_ready
            if not self.contact_valid:
                self.candidate_started = None
            if new_frame:
                candidate = (arm_ready and self.contact_valid and self.width_speed is not None
                             and self.width_speed <= self.policy.max_width_speed_m_s)
                if candidate:
                    if self.candidate_started is None:
                        self.candidate_started = current["timestamp_s"]
                    if current["timestamp_s"] - self.candidate_started >= self.policy.contact_dwell_s:
                        self.acquired = True
                        self.acquisitions += 1
                        self.outcome = "acquired"
                else:
                    self.candidate_started = None
        if self.acquired and self.mode in ("close", "carry"):
            if self.contact_valid:
                self.last_contact_width_m = current["width_m"]
                self.loss_started = None
            else:
                self.loss_started = now if self.loss_started is None else self.loss_started
                if now - self.loss_started >= self.policy.carry_loss_grace_s:
                    self.outcome = "grasp_lost"
                    raise GraspTaskFailure("grasp_lost", "Grasp load/width was lost; further motion blocked")
        if (self.mode == "close" and not self.acquired and arm_ready
                and self.close_active_s >= self.policy.close_timeout_s):
            self.outcome = "grasp_not_acquired"
            raise GraspTaskFailure("grasp_not_acquired", "No expected loaded contact before close timeout; lift blocked")

    @property
    def soft_gripper_exempt(self):
        return self.acquired and self.mode in ("close", "carry")

    def check_release_bridge_deadline(self, now, timeout_s):
        """Bound the opening bridge across actions using the tracking budget."""
        if (self.mode == "release" and self.release_bridge_active
                and now - self.release_started >= timeout_s):
            raise GraspFeedbackError("Release opening bridge exceeded tracking timeout")

    def release_soft_gripper_exempt(self, target_width, measured_width):
        """Allow only the initial opening targets below verified contact width.

        All feedback/hard-bound checks run before this method. This cannot
        excuse a gripper that is more closed than its opening command. Once
        the target crosses the contact width or current measured opening, the
        bridge retires permanently for this contiguous release phase.
        """
        self.release_bridge_exempt = False
        if self.mode != "release" or not self.release_bridge_active:
            return False
        if (target_width >= self.release_contact_width_m
                or measured_width <= target_width):
            self.release_bridge_active = False
            return False
        lo, hi = self.policy.expected_width_m
        # Check both the coherent gripper packet and the subsequently read
        # measured state. An out-of-window observation never expands the
        # remembered contact width or enables a soft-arrival exception.
        self.release_bridge_exempt = bool(
            self.feedback is not None and lo <= self.feedback["width_m"] <= hi
            and lo <= measured_width <= hi
        )
        return self.release_bridge_exempt

    def is_last_in_phase(self):
        return self.index + 1 == len(self.modes) or self.modes[self.index + 1] != self.mode

    def phase_wait(self):
        if not self.is_last_in_phase():
            return False
        if self.mode == "close":
            return not self.acquired
        if self.mode == "release":
            return (self.feedback is None or self.feedback["width_m"]
                    < self.policy.expected_width_m[1] + self.policy.min_closing_gap_m)
        return False

    def finish_action(self):
        if self.mode == "release" and self.is_last_in_phase():
            self.releases += 1
            self.outcome = "acquired_and_released"

    def snapshot(self):
        return {"mode": self.mode, "acquired": self.acquired, "contact_valid": self.contact_valid,
                "width_speed_m_s": self.width_speed, "candidate_started_timestamp_s": self.candidate_started,
                "loss_started_monotonic_s": self.loss_started, "acquisitions": self.acquisitions,
                "close_active_s": self.close_active_s,
                "releases": self.releases, "outcome": self.outcome, "feedback": self.feedback,
                "soft_gripper_arrival_exempt": self.soft_gripper_exempt,
                "release_contact_width_m": self.release_contact_width_m,
                "release_bridge_active": self.release_bridge_active,
                "release_bridge_started_monotonic_s": self.release_started,
                "release_gripper_arrival_exempt": self.release_bridge_exempt}
