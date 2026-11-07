# SDK unit/control conventions adapted from LeRobot's PiperMotorsBus:
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Native PiPER V2 CAN adapter, independent of LeRobot and Isaac Sim.

Reference: lerobot-piper ``src/lerobot/motors/piper/piper_motors_bus.py``
and ``piper_scripts/piper_inference/piper_client.py``. Only the SDK units,
control calls and physical limit defaults are adapted; safety/lifecycle code
is implemented here. SDK documentation: https://github.com/agilexrobotics/piper_sdk

There is no homing, automatic CAN setup, master/slave mode change or emergency
reset. Use a dedicated follower CAN connection without a competing leader.
Under the default projected policy, ``hold`` sends the measured pose and
``hold_grasp`` preserves the last applied gripper goal through close. Explicit
reference policy sends no new target during hold/close. ``stop`` requests the SDK emergency stop,
which may permit damped descent and must not be interpreted as a holding brake.
An external watchdog must call ``stop`` if the application stops servicing its
control loop. Feedback checks run on reads/commands, not in a background thread.
Stop requests latch independently of the command lock, but sending the emergency
frame must wait for an in-flight SDK call to return. This is not a hard real-time
stop mechanism and cannot cancel a blocked driver or kernel operation.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import socket
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import numpy as np


class PiperHardwareError(RuntimeError):
    """The hardware connection or feedback cannot support safe control."""


class PiperFeedbackError(PiperHardwareError):
    """Missing, stale, invalid or faulty hardware feedback."""


class _PiperPositionError(PiperFeedbackError):
    """A complete feedback sample violates a non-retryable position bound."""

    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None):
        self.diagnostics = diagnostics
        if diagnostics is not None:
            message = f"{message}; diagnostics={json.dumps(diagnostics, allow_nan=False)}"
        super().__init__(message)


class PiperProjectionError(PiperHardwareError):
    """Rejected setpoint with the exact pre-send constraint snapshot.

    Diagnostics are also embedded in the message so existing exception-only
    replay logs retain them without requiring another hardware read after stop.
    """

    def __init__(self, message: str, diagnostics: dict[str, Any]):
        self.diagnostics = diagnostics
        super().__init__(f"{message}; diagnostics={json.dumps(diagnostics, allow_nan=False)}")


def _finite_vector(value, length: int, name: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
        if np.iscomplexobj(raw) or raw.dtype.kind not in "iuf":
            raise ValueError(name)
        result = np.array(raw, dtype=np.float64, copy=True)
        if result.shape != (length,) or not np.isfinite(result).all():
            raise ValueError(name)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain {length} finite real values") from error
    return result


def _positive(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_, str)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


@dataclass(frozen=True)
class PiperHardwareConfig:
    can_name: str = "can0"
    actuation_enabled: bool = False
    control_hz: float = 30.0
    feedback_timeout_s: float = 0.25
    connect_timeout_s: float = 3.0
    enable_timeout_s: float = 3.0
    joint_lower_rad: tuple[float, ...] = tuple(np.deg2rad([-154, 0, -175, -102, -75, -170]))
    joint_upper_rad: tuple[float, ...] = tuple(np.deg2rad([154, 195, 0, 102, 75, 170]))
    max_joint_velocity_rad_s: tuple[float, ...] = (0.5,) * 6
    joint_startup_recovery_tolerance_rad: tuple[float, ...] = (0.0,) * 6
    startup_recovery_feedback_slack_rad: tuple[float, ...] = (math.radians(.002),) * 6
    startup_recovery_timeout_s: float = 10.0
    gripper_min_width_m: float = 0.0
    gripper_max_width_m: float = 0.07
    max_gripper_velocity_m_s: float = 0.02
    speed_percent: int = 10
    gripper_effort: int = 1000
    command_policy: str = "projected"

    def __post_init__(self):
        if (not isinstance(self.can_name, str) or not self.can_name
                or "/" in self.can_name or self.can_name in (".", "..")):
            raise ValueError("can_name must be an interface name")
        if type(self.command_policy) is not str or self.command_policy not in ("projected", "reference"):
            raise ValueError("command_policy must be projected or reference")
        if type(self.actuation_enabled) is not bool:
            raise ValueError("actuation_enabled must be an explicit boolean")
        for name in ("control_hz", "feedback_timeout_s", "connect_timeout_s",
                     "enable_timeout_s", "max_gripper_velocity_m_s", "startup_recovery_timeout_s"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        for name in ("joint_lower_rad", "joint_upper_rad", "max_joint_velocity_rad_s",
                     "joint_startup_recovery_tolerance_rad", "startup_recovery_feedback_slack_rad"):
            object.__setattr__(self, name, tuple(_finite_vector(getattr(self, name), 6, name)))
        if np.any(np.asarray(self.joint_lower_rad) >= self.joint_upper_rad):
            raise ValueError("joint_lower_rad must be below joint_upper_rad")
        if np.any(np.asarray(self.max_joint_velocity_rad_s) <= 0):
            raise ValueError("max_joint_velocity_rad_s must be positive")
        recovery = np.asarray(self.joint_startup_recovery_tolerance_rad)
        if np.any(recovery < 0) or np.any(recovery > math.radians(3)):
            raise ValueError("joint_startup_recovery_tolerance_rad must lie in [0, 3 degrees]")
        slack = np.asarray(self.startup_recovery_feedback_slack_rad)
        if np.any(slack < 0) or np.any(slack > math.radians(.25)):
            raise ValueError("startup_recovery_feedback_slack_rad must lie in [0, 0.25 degrees]")
        widths = _finite_vector([self.gripper_min_width_m, self.gripper_max_width_m], 2,
                                "gripper width bounds")
        if not 0 <= widths[0] < widths[1] <= 0.1:
            raise ValueError("gripper width bounds must satisfy 0 <= min < max <= 0.1 m")
        for name, lower, upper in (("speed_percent", 1, 100), ("gripper_effort", 0, 5000)):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")


def check_can_interface(can_name: str) -> None:
    """Read Linux CAN state without configuring interfaces or sending frames."""
    root = Path("/sys/class/net") / can_name
    try:
        if int((root / "type").read_text()) != 280:  # ARPHRD_CAN
            raise PiperHardwareError(f"{can_name} is not a CAN interface")
        if not int((root / "flags").read_text().strip(), 16) & 1:  # IFF_UP
            raise PiperHardwareError(f"CAN interface {can_name} is down")
        with socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW) as probe:
            probe.bind((can_name,))
    except (OSError, ValueError, AttributeError) as error:
        raise PiperHardwareError(f"CAN interface {can_name} is unavailable: {error}") from error


class PiperHardware:
    """Single-arm hardware with opt-in actuation and latched fault handling.

    By default, ``command`` rejects non-finite/out-of-range targets, projects their rate of
    change, and returns the actual quantized setpoint. Bounds apply to command
    velocity, not a measured guarantee on physical speed. ``close`` holds the
    measured pose before disconnecting, unless ``hold_grasp`` explicitly retains
    the last applied gripper goal; it never disables motors under gravity.
    ``stop`` blocks subsequent commands and does not automatically clear the
    firmware emergency stop, even after reconnecting.

    Explicit ``command_policy="reference"`` uses the original PiPER client's
    device interpolation: nominal targets are sent directly at speed_percent,
    and finite measured joint excursions do not block startup. It does not use
    software velocity slew/recovery checks. Fault, feedback freshness, target
    bounds and manual-stop handling remain. In this mode ``hold``/``close`` do
    not send a new arm/gripper target; the device retains the last applied goal.

    Startup recovery is an explicit, bounded exception for measured joints just
    outside their nominal limits. It never changes accepted action bounds. Only
    the initial excursion can be traversed, with inward, rate-limited setpoints;
    ordinary recovery retires each allowance on first nominal entry. Explicit
    native recovery retains only its original J2/J3 feedback slack until stable
    inset arrival, then retires the allowances together.
    Recovery does not move automatically and its timeout is checked on reads.
    """

    def __init__(self, config: PiperHardwareConfig | Mapping[str, Any] | None = None, *,
                 sdk_factory: Callable | None = None, can_probe: Callable = check_can_interface,
                 monotonic: Callable = time.monotonic, wall_time: Callable = time.time,
                 sleep: Callable = time.sleep):
        self.config = (config if isinstance(config, PiperHardwareConfig)
                       else PiperHardwareConfig(**dict(config or {})))
        self._factory = sdk_factory
        self._can_probe = can_probe
        self._monotonic, self._wall_time, self._sleep = monotonic, wall_time, sleep
        self._lock = threading.RLock()
        self._stop_requested = threading.Event()
        self._sdk = None
        self._can_bus = self._original_send = self._original_callback = None
        self._frame_lock = threading.Lock()
        self._frames: dict[int, tuple[float, float]] = {}
        self._gripper_frame: tuple[float, float, bytes] | None = None
        self._owns_sdk = False
        self._connected = self._enabled = self._stopped = self._stop_sent = False
        self._last_error: str | None = None
        self._feedback_stamps: dict[str, float] = {}
        self._feedback_updates: dict[str, float] = {}
        self._last_state: np.ndarray | None = None
        self._last_command: np.ndarray | None = None
        self._last_command_at: float | None = None
        self._grasp_hold_target_m: float | None = None
        self._lower = np.r_[self.config.joint_lower_rad, self.config.gripper_min_width_m]
        self._upper = np.r_[self.config.joint_upper_rad, self.config.gripper_max_width_m]
        if self.config.command_policy == "reference":
            # A custom configuration can narrow the reference client's nominal
            # ranges, never expand them, including direct adapter command use.
            self._lower = np.maximum(self._lower, np.r_[
                PiperHardwareConfig.joint_lower_rad, PiperHardwareConfig.gripper_min_width_m])
            self._upper = np.minimum(self._upper, np.r_[
                PiperHardwareConfig.joint_upper_rad, PiperHardwareConfig.gripper_max_width_m])
            if np.any(self._lower >= self._upper):
                raise ValueError("Reference and configured hardware target ranges must overlap")
        self._velocity = np.r_[self.config.max_joint_velocity_rad_s,
                               self.config.max_gripper_velocity_m_s]
        self._sdk_scale = np.r_[np.full(6, 180000.0 / np.pi), 1_000_000.0]
        self._recovery_pending = False
        self._recovery_direction = np.zeros(6, dtype=np.int8)
        self._native_recovery_retained = np.zeros(6, dtype=bool)
        self._native_check_inward_progress = True
        self._recovery_initial_state: np.ndarray | None = None
        self._recovery_best_excursion = np.zeros(6)
        self._recovery_started_at: float | None = None
        # This allowance applies only to measured feedback. Sent recovery
        # references retain their original initial envelope and SDK quantization.
        self._recovery_feedback_slack = np.asarray(self.config.startup_recovery_feedback_slack_rad)

    def connect(self, *, expected_state=None, expected_tolerance=None, cancel_event=None) -> None:
        """Connect, optionally checking an operator-reviewed pose before enable.

        Both optional arguments are finite SI vectors of length seven, with
        strictly positive tolerances. Invalid arguments fail before CAN access;
        a pose mismatch closes the passive connection without enabling motors.
        An optional event with ``is_set()`` also cancels feedback waiting and is
        checked before and after enable operations. It cannot interrupt a
        blocking SDK call; cancellation during enabling requests a stop.
        """
        if (expected_state is None) != (expected_tolerance is None):
            raise ValueError("expected_state and expected_tolerance must be supplied together")
        if expected_state is not None:
            expected_state = _finite_vector(expected_state, 7, "expected_state")
            expected_tolerance = _finite_vector(expected_tolerance, 7, "expected_tolerance")
            if np.any(expected_tolerance <= 0):
                raise ValueError("expected_tolerance must be positive")
        if cancel_event is not None and not callable(getattr(cancel_event, "is_set", None)):
            raise ValueError("cancel_event must provide is_set()")
        self._check_connect_cancel(cancel_event)
        with self._lock:
            if self._connected:
                raise PiperHardwareError("PiPER is already connected")
            self._last_error = None
            self._stopped = self._stop_sent = False
            self._stop_requested.clear()
            self._feedback_stamps.clear()
            self._feedback_updates.clear()
            with self._frame_lock:
                self._frames.clear()
                self._gripper_frame = None
            self._last_state = self._last_command = self._last_command_at = None
            self._grasp_hold_target_m = None
            self._recovery_pending = True
            self._recovery_direction.fill(0)
            self._native_recovery_retained.fill(False)
            self._native_check_inward_progress = True
            self._recovery_initial_state = self._recovery_started_at = None
            self._recovery_best_excursion.fill(0)
            try:
                self._can_probe(self.config.can_name)
                factory = self._factory
                if factory is None:
                    try:
                        from piper_sdk import C_PiperInterface_V2
                    except ImportError as error:
                        raise PiperHardwareError("Install the optional piper_sdk package for real hardware") from error
                    factory = C_PiperInterface_V2
                self._sdk = factory(self.config.can_name)
                if self._sdk.get_connect_status():
                    raise PiperHardwareError("PiPER SDK connection is already owned by another controller")
                self._owns_sdk = True
                self._install_checked_transport()
                # piper_init=False avoids implicit parameter changes/initialization.
                self._sdk.ConnectPort(piper_init=False)
                if not self._sdk.get_connect_status():
                    raise PiperHardwareError("PiPER SDK failed to connect the CAN port")
                self._connected = True
                self._await_feedback(cancel_event=cancel_event)
                self._check_connect_cancel(cancel_event)
                self._check_expected_state(self._last_state, expected_state, expected_tolerance)
                if self.config.actuation_enabled:
                    self._await_enabled(expected_state, expected_tolerance, cancel_event)
            except Exception as error:
                self._last_error = str(error)
                if self._enabled:
                    self._stop_no_raise()
                self._disconnect_no_raise()
                raise

    def _install_checked_transport(self) -> None:
        """Turn SDK log-only transmission errors into exceptions on our bus.

        V2 motion methods return None even after failed sends. GetCanBus is the
        SDK's public transport accessor; the connection is exclusively owned,
        and its original sender is restored during disconnect. Acceptance by
        SocketCAN is not a physical acknowledgement that the robot has stopped.
        """
        try:
            bus = self._sdk.GetCanBus()
            original = bus.SendCanMessage
            callback = bus.callback_function
            if not callable(callback):
                raise AttributeError("CAN receive callback is unavailable")
            success = bus.CAN_STATUS.SEND_MESSAGE_SUCCESS
        except AttributeError as error:
            raise PiperHardwareError("PiPER requires a SDK with checked GetCanBus transport and receive callback support") from error

        def checked_send(can_id, data, *args, **kwargs):
            # Serialize SDK encoding: its shared tx_msg is not safe for a
            # concurrent emergency command. An independent latch prevents the
            # remaining command frames once an in-flight send returns.
            emergency = can_id == 0x150 and len(data) > 0 and data[0] == 0x01
            if self._stop_requested.is_set() and not emergency:
                raise PiperHardwareError("PiPER stop requested; command transmission interrupted")
            result = original(can_id, data, *args, **kwargs)
            if result != success:
                raise PiperHardwareError(f"PiPER CAN transmission failed for 0x{can_id:03x}: {result}")
            if self._stop_requested.is_set() and not emergency:
                raise PiperHardwareError("PiPER stop requested during CAN transmission")
            return result

        def checked_receive(message):
            callback(message)
            # The SDK's joint/driver timestamps describe the latest aggregate
            # frame. Observe every required frame to reject default zero fields
            # and a lost CAN pair even while its peers continue publishing.
            if (getattr(message, "is_error_frame", False) or getattr(message, "is_remote_frame", False)
                    or getattr(message, "is_extended_id", False)
                    or getattr(message, "dlc", 0) != 8 or len(getattr(message, "data", ())) != 8):
                return
            try:
                timestamp = float(message.timestamp)
                can_id = int(message.arbitration_id)
            except (AttributeError, TypeError, ValueError):
                return
            if math.isfinite(timestamp) and timestamp > 0:
                with self._frame_lock:
                    received = self._monotonic()
                    self._frames[can_id] = (timestamp, received)
                    if can_id == 0x2A8:
                        self._gripper_frame = (timestamp, received, bytes(message.data))

        self._can_bus, self._original_send, self._original_callback = bus, original, callback
        bus.SendCanMessage = checked_send
        bus.callback_function = checked_receive

    def _check_frames(self) -> None:
        required = [0x2A5, 0x2A6, 0x2A7, 0x2A8]
        if self.config.actuation_enabled:
            required += [0x2A1, *range(0x261, 0x267)]
        now, mono = self._wall_time(), self._monotonic()
        with self._frame_lock:
            frames = dict(self._frames)
        for can_id in required:
            sample = frames.get(can_id)
            if sample is None:
                raise PiperFeedbackError(f"Missing PiPER feedback frame 0x{can_id:03x}")
            stamp, received = sample
            if (now - stamp > self.config.feedback_timeout_s or stamp - now > .05
                    or mono - received > self.config.feedback_timeout_s):
                raise PiperFeedbackError(f"Stale PiPER feedback frame 0x{can_id:03x}")

    @staticmethod
    def _check_connect_cancel(cancel_event) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise PiperHardwareError("PiPER connection cancelled")

    def _await_feedback(self, *, cancel_event=None) -> None:
        deadline = self._monotonic() + self.config.connect_timeout_s
        first_stamps = None
        detail = "no feedback received"
        while self._monotonic() < deadline:
            self._check_connect_cancel(cancel_event)
            try:
                self._read_feedback()
                stamps = dict(self._feedback_stamps)
                if first_stamps is None:
                    first_stamps = stamps
                elif all(stamps[key] > value for key, value in first_stamps.items()):
                    return
                detail = "feedback timestamps are not advancing"
            except _PiperPositionError:
                raise
            except PiperFeedbackError as error:
                detail = str(error)
            self._sleep(0.01)
        raise PiperFeedbackError(f"Timed out waiting for live PiPER feedback: {detail}")

    def _stamp(self, name: str, message) -> None:
        try:
            stamp = float(message.time_stamp)
        except (AttributeError, ValueError, TypeError) as error:
            raise PiperFeedbackError(f"Missing {name} feedback timestamp") from error
        now, mono = self._wall_time(), self._monotonic()
        previous = self._feedback_stamps.get(name)
        if (not math.isfinite(stamp) or stamp <= 0 or now - stamp > self.config.feedback_timeout_s
                or stamp - now > 0.05 or (previous is not None and stamp < previous)):
            raise PiperFeedbackError(f"Invalid or stale {name} feedback timestamp")
        if previous != stamp:
            self._feedback_updates[name] = mono
        elif mono - self._feedback_updates[name] > self.config.feedback_timeout_s:
            raise PiperFeedbackError(f"Stale {name} feedback (timestamp stopped advancing)")
        self._feedback_stamps[name] = stamp

    def _check_arm_status(self) -> None:
        message = self._sdk.GetArmStatus()
        self._stamp("arm_status", message)
        status = message.arm_status
        # err_code is the raw SDK bitfield; err_status is its decoded object.
        if int(status.arm_status) != 0 or int(status.err_code) != 0:
            raise PiperFeedbackError(f"PiPER reports arm_status={status.arm_status}, err_code={status.err_code}")

    def _read_feedback(self, *, check_enabled: bool = True) -> np.ndarray:
        if not self._connected or not self._sdk.get_connect_status():
            raise PiperHardwareError("PiPER is not connected")
        joints = self._sdk.GetArmJointMsgs()
        self._stamp("joints", joints)
        # Copy immediately: SDK returns mutable cached message objects.
        values = [getattr(joints.joint_state, f"joint_{index}") for index in range(1, 7)]
        gripper = self._sdk.GetArmGripperMsgs()
        self._stamp("gripper", gripper)
        values.append(gripper.gripper_state.grippers_angle)
        try:
            state = _finite_vector(values, 7, "PiPER feedback") / self._sdk_scale
        except ValueError as error:
            raise PiperFeedbackError(str(error)) from error
        if state[6] < self._lower[6] - 1e-7 or state[6] > self._upper[6] + 1e-7:
            raise PiperFeedbackError("PiPER feedback is outside configured joint/gripper limits")
        if self.config.actuation_enabled:
            self._check_arm_status()
            enabled = self._drivers_enabled()
            if check_enabled and self._enabled and not enabled:
                raise PiperFeedbackError("PiPER motor enable feedback was lost")
        self._check_frames()
        self._check_joint_feedback_bounds(state)
        self._last_state = state.copy()
        return state

    def _check_joint_feedback_bounds(self, state: np.ndarray) -> None:
        """Capture one startup envelope, then shrink it from real feedback.

        Normal measurements retain the existing strict limits. A recovering
        joint may settle within its configured feedback slack (two encoder units
        by default, at most 0.25 degrees) around its best inward sample. The
        fixed initial excursion plus slack and configured startup tolerance also
        cap feedback; the allowance never grows after capture. This does not
        widen commanded setpoints. Ordinary nominal entry retires the exception.
        Explicit native recovery can opt out of monotonic progress checks.
        Only its original recovering J2/J3 then use the fixed configured startup
        tolerance (at most 3 degrees), until stable inset arrival or scope exit.
        Ordinary recovery and the default native policy keep shrinking bounds.
        """
        # The reference client bounds commanded targets, not measured joint
        # feedback. Device faults/freshness remain checked by _read_feedback.
        # Do not synthesize a startup recovery phase or widen target limits.
        if self.config.command_policy == "reference":
            self._recovery_pending = False
            return
        q, lower, upper = state[:6], self._lower[:6], self._upper[:6]
        below, above = q < lower - 1e-7, q > upper + 1e-7
        excursion = np.maximum(lower - q, np.maximum(q - upper, 0))

        def reject(message, offending, allowed=None):
            raise _PiperPositionError(message, dict(
                schema_version=1, event="feedback_bounds_rejected",
                offending_indices=np.flatnonzero(offending).tolist(), measured_si=state.tolist(),
                joint_lower_rad=lower.tolist(), joint_upper_rad=upper.tolist(),
                recovery_direction=self._recovery_direction.tolist(),
                native_retained_indices=np.flatnonzero(self._native_recovery_retained).tolist(),
                check_inward_progress=self._native_check_inward_progress,
                best_excursion_rad=self._recovery_best_excursion.tolist(),
                allowed_excursion_rad=None if allowed is None else np.asarray(allowed).tolist(),
                feedback_timestamps=dict(self._feedback_stamps)))

        if self._recovery_pending:
            tolerance = np.asarray(self.config.joint_startup_recovery_tolerance_rad)
            if np.any(excursion > tolerance + 1e-7):
                reject("PiPER feedback is outside configured joint/gripper limits",
                       excursion > tolerance + 1e-7, tolerance)
            self._recovery_pending = False
            self._recovery_initial_state = state.copy()
            self._recovery_direction = above.astype(np.int8) - below.astype(np.int8)
            self._recovery_best_excursion = excursion.copy()
            if np.any(self._recovery_direction):
                self._recovery_started_at = self._monotonic()
        active = self._recovery_direction != 0
        if (active.any() and self._monotonic() - self._recovery_started_at
                > self.config.startup_recovery_timeout_s):
            reject("PiPER startup recovery timed out", active)
        if np.any((below | above) & ~active):
            reject("PiPER feedback is outside configured joint/gripper limits", (below | above) & ~active)
        opposite = ((below & (self._recovery_direction > 0))
                    | (above & (self._recovery_direction < 0)))
        if np.any(opposite):
            reject("PiPER startup recovery crossed the opposite joint limit", opposite)
        if not active.any():
            return
        initial_q = self._recovery_initial_state[:6]
        initial_excursion = np.maximum(lower - initial_q, np.maximum(initial_q - upper, 0))
        allowed_excursion = np.minimum(
            self.config.joint_startup_recovery_tolerance_rad,
            np.minimum(initial_excursion, self._recovery_best_excursion) + self._recovery_feedback_slack)
        bounded_native = (active & self._native_recovery_retained
                          if not self._native_check_inward_progress else np.zeros(6, dtype=bool))
        allowed_excursion[bounded_native] = np.asarray(
            self.config.joint_startup_recovery_tolerance_rad)[bounded_native]
        if np.any(active & (excursion > allowed_excursion + 1e-12)):
            reject(("PiPER native startup feedback exceeds configured startup tolerance"
                    if np.any(bounded_native & (excursion > allowed_excursion + 1e-12))
                    else "PiPER startup recovery feedback moved outward"),
                   active & (excursion > allowed_excursion + 1e-12), allowed_excursion)
        self._recovery_best_excursion[active] = np.minimum(
            self._recovery_best_excursion[active], excursion[active])
        self._recovery_direction[~(below | above) & ~self._native_recovery_retained] = 0

    def _drivers_enabled(self) -> bool:
        message = self._sdk.GetArmLowSpdInfoMsgs()
        self._stamp("motor_status", message)
        statuses = [getattr(message, f"motor_{index}").foc_status for index in range(1, 7)]
        fault_fields = ("voltage_too_low", "motor_overheating", "driver_overcurrent",
                        "driver_overheating", "collision_status", "driver_error_status", "stall_status")
        if any(getattr(status, name) for status in statuses for name in fault_fields):
            raise PiperFeedbackError("PiPER motor driver reports a fault")
        gripper = self._sdk.GetArmGripperMsgs()
        self._stamp("gripper", gripper)
        gripper_status = gripper.gripper_state.foc_status
        gripper_fault_fields = ("voltage_too_low", "motor_overheating", "driver_overcurrent",
                                "driver_overheating", "sensor_status", "driver_error_status")
        if any(getattr(gripper_status, name) for name in gripper_fault_fields):
            raise PiperFeedbackError("PiPER gripper driver reports a fault")
        return (all(status.driver_enable_status for status in statuses)
                and bool(gripper_status.driver_enable_status))

    @staticmethod
    def _check_expected_state(state, expected_state, expected_tolerance) -> None:
        if (expected_state is not None
                and np.any(np.abs(state - expected_state) > expected_tolerance)):
            raise PiperHardwareError("PiPER initial state differs from the reviewed expected_state")

    def _await_enabled(self, expected_state=None, expected_tolerance=None, cancel_event=None) -> None:
        deadline = self._monotonic() + self.config.enable_timeout_s
        while self._monotonic() < deadline:
            self._check_connect_cancel(cancel_event)
            state = self._read_feedback(check_enabled=False)
            self._check_expected_state(state, expected_state, expected_tolerance)
            self._check_connect_cancel(cancel_event)
            # Treat partial enabling as active so a timeout requests a stop,
            # but a failed pre-enable pose check still sends no control frames.
            self._enabled = True
            self._sdk.EnableArm(7)
            self._check_connect_cancel(cancel_event)
            # Preserve actual opening; the reference client's zero would close it.
            width = int(round(state[6] * 1_000_000))
            self._sdk.GripperCtrl(width, self.config.gripper_effort, 0x01, 0)
            enabled = self._drivers_enabled()
            self._check_connect_cancel(cancel_event)
            if enabled:
                return
            self._sleep(0.01)
        raise PiperHardwareError("Timed out waiting for all six motors and gripper to enable")

    def read_state(self) -> np.ndarray:
        with self._lock:
            try:
                return self._read_feedback()
            except Exception as error:
                self._last_error = str(error)
                self._stop_no_raise()
                raise

    def read_gripper_feedback(self) -> dict[str, Any]:
        """Read one coherent gripper CAN frame without sending any commands.

        Returns a defensive dictionary containing ``width_m`` (full opening),
        signed ``effort_raw`` (SDK unit: 0.001 N·m, not force or motor current),
        ``status_raw``, ``enabled``, ``fault``, and source ``timestamp_s`` (Unix
        seconds). Status bits 0–5 indicate faults, bit 6 enables the driver and
        bit 7 reports homing; homing is not a fault. A reported fault is returned
        as ``fault=True`` for the caller to handle, without issuing a stop here.

        All returned fields are decoded from one immutable received 0x2A8 frame.
        The SDK getter returns a mutable shared object after releasing its lock;
        it is used only for separate schema/freshness validation. Its fields are
        never combined with a different raw frame. Missing, unsupported, invalid
        or stale feedback raises ``PiperFeedbackError`` without changing control
        state or sending CAN frames. Repeated reads may return the same still
        fresh timestamp; callers must require timestamp advances for dwell tests.
        """
        with self._lock:
            if not self._connected or not self._sdk.get_connect_status():
                raise PiperHardwareError("PiPER is not connected")
            try:
                message = self._sdk.GetArmGripperMsgs()
                state = message.gripper_state
                sdk_values = (state.grippers_angle, state.grippers_effort, state.status_code)
            except AttributeError as error:
                raise PiperFeedbackError("PiPER SDK does not expose complete gripper feedback") from error
            for value, name, lower, upper in zip(
                sdk_values, ("grippers_angle", "grippers_effort", "status_code"),
                (-(2**31), -(2**15), 0), (2**31 - 1, 2**15 - 1, 255),
            ):
                if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                        or not lower <= value <= upper):
                    raise PiperFeedbackError(f"PiPER SDK reports invalid gripper {name}")
            self._stamp("gripper_feedback_sdk", message)
            with self._frame_lock:
                snapshot = self._gripper_frame
            if snapshot is None:
                raise PiperFeedbackError("Missing PiPER gripper feedback frame 0x2a8")
            stamp, received, payload = snapshot
            self._stamp("gripper_feedback_frame", SimpleNamespace(time_stamp=stamp))
            if self._monotonic() - received > self.config.feedback_timeout_s:
                raise PiperFeedbackError("Stale PiPER gripper feedback frame 0x2a8")
            width = int.from_bytes(payload[:4], "big", signed=True) / 1_000_000.
            if not self._lower[6] <= width <= self._upper[6]:
                raise PiperFeedbackError("PiPER gripper feedback is outside configured width limits")
            effort = int.from_bytes(payload[4:6], "big", signed=True)
            status = payload[6]
            return dict(width_m=width, effort_raw=effort, status_raw=status,
                        enabled=bool(status & 0x40), fault=bool(status & 0x3F), timestamp_s=stamp)

    def validate_target(self, target) -> np.ndarray:
        """Validate absolute SI targets without issuing CAN commands."""
        value = _finite_vector(target, 7, "PiPER target")
        # Runtime ABI actions are float32; a representable boundary may lie a
        # fraction of a micro-unit beyond the original float64 configuration.
        # Match the shared action contract, then restore exact physical bounds.
        if np.any(value < self._lower - 1e-6) or np.any(value > self._upper + 1e-6):
            raise ValueError("PiPER target is outside configured joint/gripper limits")
        return np.clip(value, self._lower, self._upper)

    def _require_actuation(self) -> None:
        if not self.config.actuation_enabled:
            raise PiperHardwareError("PiPER actuation is disabled; set actuation_enabled=true explicitly")
        if not self._connected or not self._enabled:
            raise PiperHardwareError("PiPER is not connected and enabled")
        if self._stopped or self._stop_requested.is_set():
            raise PiperHardwareError("PiPER stop is latched; operator recovery and reconnect are required")

    def _send(self, raw: np.ndarray) -> np.ndarray:
        try:
            self._sdk.MotionCtrl_2(0x01, 0x01, self.config.speed_percent, 0x00)
            self._sdk.JointCtrl(*(int(value) for value in raw[:6]))
            self._sdk.GripperCtrl(int(raw[6]), self.config.gripper_effort, 0x01, 0)
            if self._stop_requested.is_set():
                raise PiperHardwareError("PiPER stop requested; command acknowledgement rejected")
        except Exception as error:
            self._last_error = str(error)
            self._stop_no_raise()
            raise
        applied = raw / self._sdk_scale
        self._last_command = applied.copy()
        self._last_command_at = self._monotonic()
        return applied.copy()

    def command(self, target) -> np.ndarray:
        with self._lock:
            self._grasp_hold_target_m = None
            self._require_actuation()
            try:
                target = self.validate_target(target)
            except ValueError as error:
                self._last_error = str(error)
                self._stop_no_raise()
                raise
            measured = self.read_state()
            desired = np.rint(target * self._sdk_scale)
            # Quantize the goal into nominal bounds before rate projection.
            # Clamping the projected result would jump from an outside start to
            # the boundary, bypassing the configured velocity limit.
            desired = np.clip(desired, np.ceil(self._lower * self._sdk_scale),
                              np.floor(self._upper * self._sdk_scale))
            if self.config.command_policy == "reference":
                # Match PiperMotorsBus: one nominal endpoint with MOVEJ device
                # interpolation at speed_percent. A software slew baseline
                # outside bounds must never become an intermediate setpoint.
                return self._send(desired)
            return self._send_projected(desired, measured)

    def recover_startup_native(
        self, *, verified_max_joint_speed_rad_s, speed_percent: int = 1,
        inset_rad: float = math.radians(1), max_travel_rad: float = math.radians(4),
        measured_speed_limit_rad_s: float = math.radians(5),
        nonrecover_drift_rad: float = math.radians(1), timeout_s: float = 20.,
        stable_s: float = .5, tolerance_rad: float = math.radians(.25),
        check_inward_progress: bool = True,
        cancel_event=None, progress_callback=None,
    ) -> dict[str, Any]:
        """Explicit device-planned recovery of a small initial J2/J3 excursion.

        Call after ``connect`` and before any ordinary command. Fresh feedback
        defines one complete, nominal joint target: recovering J2/J3 are inset
        from their limits, and other joints/gripper retain their measured pose.
        This sends a legal MOVEJ endpoint at 1% device speed; its first reference
        deliberately does not satisfy the ordinary setpoint slew limit.

        The caller must verify the six device maximum speeds and their units
        against the installed firmware before supplying them here. This method
        does not query/change firmware parameters, enable, or reset an emergency
        stop. The theoretical 1% speeds must be below the measured speed guard.
        Faults, freshness and startup deadlines remain enforced. By default,
        original recovering J2/J3 retain shrinking feedback slack until stable
        inset arrival. ``check_inward_progress=False`` disables only monotonic
        feedback/reversal checks during this call, using their fixed configured
        startup tolerance (at most 3 degrees) instead. It does not change checks
        during connect/enable, reopen retired joints, or change target limits.
        Travel, measured speed, other-axis drift and gripper drift (1 mm) remain
        bounded under both policies.

        Measured joint speed is total observed travel over at least 0.2 seconds
        of each joint-pair's CAN timestamps, not differences between cached reads.
        Progress callbacks receive JSON-compatible snapshots only after all
        position frame timestamps advance. Success requires a fresh stable pose,
        sends that measured nominal pose as a hold, and rebases ordinary control.
        Failure latches stop. An external watchdog remains required for blocked
        SDK calls; the same stop latency limitations as ``stop`` apply.
        """
        if self.config.command_policy != "projected":
            raise PiperHardwareError("Native startup recovery requires projected command policy")
        verified = _finite_vector(verified_max_joint_speed_rad_s, 6,
                                  "verified_max_joint_speed_rad_s")
        if type(check_inward_progress) is not bool:
            raise ValueError("check_inward_progress must be an explicit boolean")
        if np.any(verified <= 0):
            raise ValueError("verified_max_joint_speed_rad_s must be positive")
        if type(speed_percent) is not int or speed_percent != 1:
            raise ValueError("native startup recovery speed_percent must be integer 1")
        options = dict(inset_rad=inset_rad, max_travel_rad=max_travel_rad,
                       measured_speed_limit_rad_s=measured_speed_limit_rad_s,
                       nonrecover_drift_rad=nonrecover_drift_rad, timeout_s=timeout_s,
                       stable_s=stable_s, tolerance_rad=tolerance_rad)
        caps = dict(inset_rad=math.radians(1), max_travel_rad=math.radians(4),
                    measured_speed_limit_rad_s=math.radians(5),
                    nonrecover_drift_rad=math.radians(1), timeout_s=20.,
                    stable_s=20., tolerance_rad=math.radians(.25))
        for name, value in options.items():
            options[name] = _positive(value, name)
            if options[name] > caps[name]:
                raise ValueError(f"native startup recovery {name} exceeds its safety cap")
        inset_rad, max_travel_rad = options["inset_rad"], options["max_travel_rad"]
        measured_speed_limit_rad_s = options["measured_speed_limit_rad_s"]
        nonrecover_drift_rad, timeout_s = options["nonrecover_drift_rad"], options["timeout_s"]
        stable_s, tolerance_rad = options["stable_s"], options["tolerance_rad"]
        if not .5 <= stable_s < timeout_s:
            raise ValueError("native stable_s must be at least 0.5 and below timeout_s")
        if not tolerance_rad < inset_rad < max_travel_rad:
            raise ValueError("native tolerance_rad < inset_rad < max_travel_rad is required")
        if np.any(verified * .01 >= measured_speed_limit_rad_s):
            raise ValueError("verified 1% joint speeds must be below the measured speed guard")
        if cancel_event is not None and not callable(getattr(cancel_event, "is_set", None)):
            raise ValueError("cancel_event must provide is_set()")
        if progress_callback is not None and not callable(progress_callback):
            raise ValueError("progress_callback must be callable")

        deadline = None

        def check_cancel():
            self._require_actuation()
            if cancel_event is not None and cancel_event.is_set():
                raise PiperHardwareError("PiPER native startup recovery cancelled")
            if deadline is not None and self._monotonic() > deadline:
                raise PiperHardwareError("PiPER native startup recovery timed out")

        def position_stamps():
            with self._frame_lock:
                return np.array([self._frames[can_id][0]
                                 for can_id in (0x2A5, 0x2A6, 0x2A7, 0x2A8)])

        def send_nominal(value):
            # All targets including the final measured hold remain nominal.
            value = self.validate_target(value)
            raw = np.clip(np.rint(value * self._sdk_scale),
                          np.ceil(self._lower * self._sdk_scale),
                          np.floor(self._upper * self._sdk_scale))
            check_cancel()
            self._sdk.MotionCtrl_2(0x01, 0x01, 1, 0x00)
            check_cancel()
            self._sdk.JointCtrl(*(int(v) for v in raw[:6]))
            check_cancel()
            self._sdk.GripperCtrl(int(raw[6]), self.config.gripper_effort, 0x01, 0)
            check_cancel()
            return raw / self._sdk_scale

        # The independent stop latch is checked on every iteration and between
        # SDK sends. The lock prevents another controller changing the target.
        with self._lock:
            self._require_actuation()
            if self._last_command is not None:
                raise PiperHardwareError("Native startup recovery must precede ordinary commands")
            if self._native_recovery_retained.any():
                raise PiperHardwareError("Native startup recovery is already active")
            try:
                check_cancel()
                # Capture only still-active exceptions before the fresh read:
                # that read itself may be the first nominal boundary sample.
                original_active = self._recovery_direction != 0
                if np.any(original_active[[0, 3, 4, 5]]):
                    raise PiperHardwareError("Native startup recovery permits only initial J2/J3 excursions")
                self._native_recovery_retained[:] = original_active
                self._native_check_inward_progress = check_inward_progress
                start = self.read_state()
                direction = self._recovery_direction.copy()
                active = direction != 0
                if np.any(active[[0, 3, 4, 5]]):
                    raise PiperHardwareError("Native startup recovery permits only initial J2/J3 excursions")
                target = start.copy()
                target[:6][direction < 0] = self._lower[:6][direction < 0] + inset_rad
                target[:6][direction > 0] = self._upper[:6][direction > 0] - inset_rad
                target = self.validate_target(target)
                if np.any(active & (direction * (target[:6] - start[:6]) > 1e-12)):
                    raise PiperHardwareError("Native startup recovery goal would reverse inward progress")
                if np.any(np.abs(target[:6] - start[:6]) > max_travel_rad + 1e-12):
                    raise PiperHardwareError("Native startup recovery goal exceeds maximum travel")
                started = self._monotonic()
                deadline = started + timeout_s
                stamps = position_stamps()
                histories = [deque([(stamps[i], start[2*i:2*i+2].copy())]) for i in range(3)]
                best_progress = np.zeros(6)
                stable_since = None
                measured_speed = None
                last_progress_stamps = stamps.copy()

                def snapshot(state, stable_elapsed=0., completed=False):
                    return dict(state=state.tolist(), target=target.tolist(),
                                check_inward_progress=check_inward_progress,
                                elapsed_s=float(self._monotonic() - started),
                                measured_speed_rad_s=(None if measured_speed is None
                                                      else measured_speed.tolist()),
                                recovery_joint_indices=np.flatnonzero(active).tolist(),
                                stable_elapsed_s=float(stable_elapsed), completed=completed,
                                feedback_timestamps={f"0x{can_id:03x}": float(stamp) for can_id, stamp
                                    in zip((0x2A5, 0x2A6, 0x2A7, 0x2A8), stamps)})

                def check_motion(state, stamps):
                    nonlocal best_progress, measured_speed
                    check_cancel()
                    travel = np.abs(state[:6] - start[:6])
                    if np.any(travel > max_travel_rad + 1e-12):
                        raise PiperHardwareError("PiPER native startup recovery exceeded total travel")
                    if np.any(travel[~active] > nonrecover_drift_rad + 1e-12):
                        raise PiperHardwareError("PiPER native startup recovery nonrecovering joint drift")
                    if abs(state[6] - start[6]) > .001 + 1e-12:
                        raise PiperHardwareError("PiPER native startup recovery gripper drift exceeds 1 mm")
                    progress = -direction * (state[:6] - start[:6])
                    if check_inward_progress and np.any(
                            active & (progress < best_progress - self._recovery_feedback_slack - 1e-12)):
                        raise PiperHardwareError("PiPER native startup recovery reversed inward progress")
                    best_progress = np.maximum(best_progress, progress)
                    speeds = []
                    for pair, history in enumerate(histories):
                        stamp = stamps[pair]
                        if stamp > history[-1][0]:
                            history.append((stamp, state[2*pair:2*pair+2].copy()))
                        while len(history) > 2 and stamp - history[1][0] >= .2:
                            history.popleft()
                        dt = stamp - history[0][0]
                        if dt >= .2:
                            changes = np.diff(np.stack([sample[1] for sample in history]), axis=0)
                            speed = np.abs(changes).sum(axis=0) / dt
                            if np.any(speed > measured_speed_limit_rad_s + 1e-12):
                                raise PiperHardwareError("PiPER native startup recovery measured speed exceeded")
                            speeds.extend(speed)
                    measured_speed = np.array(speeds) if len(speeds) == 6 else None

                def arrived(state):
                    nominal = np.all(state >= self._lower) and np.all(state <= self._upper)
                    return (nominal and np.all(np.abs(state[:6] - target[:6]) <= tolerance_rad)
                            and abs(state[6] - target[6]) <= .001)

                if not active.any():
                    self._last_command, self._last_command_at = start.copy(), self._monotonic()
                    return snapshot(start, completed=True)
                send_nominal(target)
                while True:
                    self._sleep(min(.02, 1. / self.config.control_hz))
                    check_cancel()
                    state = self.read_state()
                    stamps = position_stamps()
                    check_motion(state, stamps)
                    # Cached snapshots do not renew a watchdog heartbeat or
                    # count toward stable arrival, even if wall time advances.
                    if not np.all(stamps > last_progress_stamps):
                        continue
                    last_progress_stamps = stamps.copy()
                    feedback_time = float(min(stamps))
                    if arrived(state):
                        if stable_since is None:
                            stable_since = feedback_time
                    else:
                        stable_since = None
                    stable_elapsed = 0. if stable_since is None else feedback_time - stable_since
                    completed = stable_elapsed >= stable_s and measured_speed is not None
                    update = snapshot(state, stable_elapsed)
                    if progress_callback is not None:
                        progress_callback(update)
                    check_cancel()
                    if completed:
                        # Logging/heartbeat callbacks can take time. Re-read
                        # before holding so a stale snapshot cannot pull back
                        # the arm or conceal a newly violated motion bound.
                        state = self.read_state()
                        stamps = position_stamps()
                        check_motion(state, stamps)
                        if not arrived(state):
                            stable_since = None
                            send_nominal(target)
                            continue
                        applied = send_nominal(state)
                        self._last_command, self._last_command_at = applied.copy(), self._monotonic()
                        self._recovery_direction.fill(0)
                        self._recovery_best_excursion.fill(0)
                        self._recovery_initial_state = self._recovery_started_at = None
                        return snapshot(state, stable_elapsed, completed=True)
                    send_nominal(target)
            except Exception as error:
                self._last_error = str(error)
                self._stop_no_raise()
                raise
            finally:
                # Scope exit restores ordinary first-entry retirement even
                # after a failure whose last tolerated sample was outside.
                # best_excursion records whether nominal entry ever occurred.
                self._recovery_direction[self._native_recovery_retained
                                         & (self._recovery_best_excursion <= 1e-7)] = 0
                self._native_recovery_retained.fill(False)
                self._native_check_inward_progress = True

    def _send_projected(self, desired: np.ndarray, measured: np.ndarray) -> np.ndarray:
        period = 1.0 / self.config.control_hz
        if self._last_command is None:
            previous, dt = measured, period
        else:
            previous = self._last_command
            dt = min(period, max(0.0, self._monotonic() - self._last_command_at))
        baseline = np.rint(previous * self._sdk_scale)
        desired = desired.copy()
        below, above = self._recovery_direction < 0, self._recovery_direction > 0
        # While feedback is recovering, never reverse the commanded direction,
        # including when a hold sees physical lag behind the previous setpoint.
        desired[:6][below] = np.maximum(desired[:6][below], baseline[:6][below])
        desired[:6][above] = np.minimum(desired[:6][above], baseline[:6][above])
        step = np.floor(self._velocity * dt * self._sdk_scale)
        raw = baseline + np.clip(desired - baseline, -step, step)
        lower = np.ceil(self._lower * self._sdk_scale)
        upper = np.floor(self._upper * self._sdk_scale)
        if self._recovery_initial_state is not None:
            initial = np.rint(self._recovery_initial_state * self._sdk_scale)
            lower[:6][below] = np.minimum(lower[:6][below], initial[:6][below])
            upper[:6][above] = np.maximum(upper[:6][above], initial[:6][above])
            # A measured advance may overtake the last setpoint. Never pull the
            # arm back out toward that old reference. If obeying the velocity
            # cap cannot reach the shrinking envelope, fail rather than jump.
            best_lower = np.rint((self._lower[:6] - self._recovery_best_excursion)
                                  * self._sdk_scale[:6]) - 2
            best_upper = np.rint((self._upper[:6] + self._recovery_best_excursion)
                                  * self._sdk_scale[:6]) + 2
            lower[:6][below] = np.maximum(lower[:6][below], best_lower[below])
            upper[:6][above] = np.minimum(upper[:6][above], best_upper[above])
        if np.any(raw < lower) or np.any(raw > upper):
            self._last_error = "PiPER projected setpoint exceeds the active startup recovery envelope"
            # Preserve the rejected sample: post-stop feedback may already have
            # changed and cannot explain which constraint rejected this frame.
            diagnostics = dict(
                schema_version=1, event="command_projection_rejected",
                monotonic_s=float(self._monotonic()), wall_time_s=float(self._wall_time()),
                offending_indices=np.flatnonzero((raw < lower) | (raw > upper)).tolist(),
                coordinate_names=[f"joint{i}" for i in range(1, 7)] + ["gripper_width"],
                measured_si=measured.tolist(), previous_reference_si=previous.tolist(),
                baseline_raw=baseline.astype(int).tolist(), goal_raw=desired.astype(int).tolist(),
                candidate_raw=raw.astype(int).tolist(), allowed_step_raw=step.astype(int).tolist(),
                effective_lower_raw=lower.astype(int).tolist(), effective_upper_raw=upper.astype(int).tolist(),
                dt_s=float(dt), period_s=float(period),
                recovery_direction=self._recovery_direction.tolist(),
                best_excursion_rad=self._recovery_best_excursion.tolist(),
                initial_state_si=(None if self._recovery_initial_state is None
                                  else self._recovery_initial_state.tolist()),
                feedback_timestamps=dict(self._feedback_stamps),
                sdk_units=["millidegree"] * 6 + ["micrometer_full_width"])
            self._stop_no_raise()
            error = PiperProjectionError(self._last_error, diagnostics)
            self._last_error = str(error)
            raise error
        return self._send(raw)

    def hold(self) -> np.ndarray:
        """Projected: send a measured hold. Reference: retain existing goal.

        Reference mode only reads feedback and never synthesizes another target
        during cleanup. It is not a brake; use stop() for explicit cancellation.
        """
        with self._lock:
            self._grasp_hold_target_m = None
            if self._stopped or self._stop_requested.is_set():
                raise PiperHardwareError("PiPER stop is latched; hold cannot resume motion")
            was_recovering = np.any(self._recovery_direction)
            outstanding_recovery = (self._last_command is not None
                                    and (np.any(self._last_command < self._lower - 1e-7)
                                         or np.any(self._last_command > self._upper + 1e-7)))
            state = self.read_state()
            if not self.config.actuation_enabled:
                return state
            self._require_actuation()
            if self.config.command_policy == "reference":
                # The reference bus disconnects without publishing a hold pose.
                # In particular, neither an outside measurement nor a clipped
                # replacement may be sent here. Firmware keeps its last goal;
                # cancellation/faults use stop(), rather than this no-op hold.
                return state
            if was_recovering or outstanding_recovery or np.any(self._recovery_direction):
                # Finish at most the already requested small inward increment;
                # sending lagging feedback directly could reverse recovery.
                return self._send_projected(np.rint(state * self._sdk_scale), state)
            return self._send(np.rint(state * self._sdk_scale))

    def hold_grasp(self, *, gripper_target_m) -> np.ndarray:
        """Hold the measured arm pose while preserving the applied clamp goal.

        The explicit full-width argument must match the last successfully sent
        gripper target within 1 micrometre. It only confirms that target: the
        exact previous applied value is reused, without increasing effort,
        further closing, enabling, or resetting anything. Freshness, motor
        faults, position bounds and any active startup projection still apply.

        A successful call makes ``close`` preserve this same gripper target.
        An ordinary ``command`` or ``hold`` clears that choice. A fault or stop
        also clears it and prevents close from restarting motion. This preserves
        the requested clamp position; it does not establish that an object is
        securely held or override firmware emergency-stop behavior.
        """
        with self._lock:
            self._require_actuation()
            requested = _finite_vector([gripper_target_m], 1, "gripper_target_m")[0]
            if not self._lower[6] <= requested <= self._upper[6]:
                raise ValueError("gripper_target_m is outside configured width limits")
            if self._last_command is None:
                raise PiperHardwareError("Grasp hold requires a previously applied gripper target")
            previous_width = float(self._last_command[6])
            if abs(requested - previous_width) > 1e-6 + 1e-12:
                raise ValueError("gripper_target_m differs from the last applied target by more than 1 micrometre")
            was_recovering = np.any(self._recovery_direction)
            outstanding_recovery = (np.any(self._last_command < self._lower - 1e-7)
                                    or np.any(self._last_command > self._upper + 1e-7))
            measured = self.read_state()
            if self.config.command_policy == "reference":
                # Retain the already applied firmware goal without resending
                # arm feedback, which can be outside the commanded range.
                self._grasp_hold_target_m = previous_width
                return measured
            target = measured.copy()
            target[6] = previous_width
            if was_recovering or outstanding_recovery or np.any(self._recovery_direction):
                applied = self._send_projected(np.rint(target * self._sdk_scale), measured)
            else:
                applied = self._send(np.rint(target * self._sdk_scale))
            self._grasp_hold_target_m = float(applied[6])
            return applied

    def _stop_no_raise(self) -> None:
        self._grasp_hold_target_m = None
        self._stop_requested.set()
        self._stopped = True
        if self._sdk is not None and self._owns_sdk and self._enabled and not self._stop_sent:
            try:
                self._sdk.MotionCtrl_1(0x01, 0, 0)
                self._stop_sent = True
            except Exception as error:
                self._last_error = f"{self._last_error or 'Stop requested'}; emergency stop send failed: {error}"

    def stop(self) -> None:
        """Latch immediately; send emergency stop when the SDK command lock is free.

        The SDK does not expose cancellation of an in-flight operation. Keeping
        its encoder serialized avoids corrupting the emergency frame. A blocked
        SDK/driver call can therefore delay emergency transmission indefinitely.
        """
        self._stop_requested.set()
        with self._lock:
            self._stop_no_raise()
            if self._enabled and not self._stop_sent:
                raise PiperHardwareError(self._last_error or "Emergency stop could not be sent")

    def _disconnect_no_raise(self) -> None:
        try:
            if self._sdk is not None and self._owns_sdk:
                self._sdk.DisconnectPort()
        except Exception as error:
            self._last_error = f"{self._last_error or 'Disconnect requested'}; disconnect failed: {error}"
        finally:
            with self._frame_lock:
                self._gripper_frame = None
            if self._can_bus is not None and self._original_send is not None:
                self._can_bus.SendCanMessage = self._original_send
                self._can_bus.callback_function = self._original_callback
            self._can_bus = self._original_send = self._original_callback = None
            self._sdk = None
            self._owns_sdk = self._connected = self._enabled = False
            self._grasp_hold_target_m = None

    def close(self) -> None:
        with self._lock:
            failure = None
            if (self._connected and self._enabled and not self._stopped
                    and not self._stop_requested.is_set()):
                try:
                    if self._grasp_hold_target_m is None:
                        self.hold()
                    else:
                        self.hold_grasp(gripper_target_m=self._grasp_hold_target_m)
                except Exception as error:
                    failure = error
                    self._stop_no_raise()
            self._disconnect_no_raise()
            if failure is not None:
                raise PiperHardwareError(f"PiPER close failed to hold: {failure}") from failure

    def status(self) -> dict[str, Any]:
        with self._lock:
            now = self._wall_time()
            return dict(connected=self._connected, actuation_enabled=self.config.actuation_enabled,
                        motors_enabled=self._enabled, stopped=self._stopped,
                        command_policy=self.config.command_policy,
                        stop_requested=self._stop_requested.is_set(),
                        emergency_stop_sent=self._stop_sent, last_error=self._last_error,
                        startup_recovery=dict(
                            check_inward_progress=self._native_check_inward_progress,
                            active_joint_indices=np.flatnonzero(self._recovery_direction).tolist(),
                            initial_state=(None if self._recovery_initial_state is None
                                           else self._recovery_initial_state.tolist()),
                            best_excursion_rad=self._recovery_best_excursion.tolist(),
                            feedback_slack_rad=self._recovery_feedback_slack.tolist(),
                            elapsed_s=(None if self._recovery_started_at is None
                                       else max(0., self._monotonic() - self._recovery_started_at)),
                            timeout_s=self.config.startup_recovery_timeout_s),
                        feedback_age_s={name: max(0.0, now - stamp)
                                        for name, stamp in self._feedback_stamps.items()})

    def describe(self) -> dict[str, Any]:
        names = [f"joint{index}" for index in range(1, 7)] + ["gripper_width"]
        units = ["rad"] * 6 + ["m"]
        return dict(kind="piper_can_v2", robot_type="piper", num_envs=1, n_dof=7,
                    state=dict(names=names, units=units),
                    action=dict(representation="joint_position", frame="robot_base", names=names,
                                units=units, lower=self._lower.tolist(), upper=self._upper.tolist(),
                                control_hz=self.config.control_hz),
                    actuation_enabled=self.config.actuation_enabled,
                    state_format="joint6_rad_gripper_full_width_m", config=asdict(self.config),
                    sdk_joint_unit="0.001 degree", sdk_gripper_unit="0.001 mm full width",
                    stop_confirmation="SocketCAN transmission accepted; physical stop is not acknowledged",
                    stop_latency="Request latches immediately; emergency transmission waits for in-flight SDK calls; no hard deadline",
                    feedback_freshness="Individual joint-pair, gripper and driver CAN frame timestamps plus SDK timestamps",
                    command_policy=self.config.command_policy,
                    interpolation=("firmware MOVEJ at speed_percent; no adapter velocity slew guarantee"
                                   if self.config.command_policy == "reference" else
                                   "adapter setpoint slew limits followed by firmware MOVEJ"),
                    measured_joint_bounds_enforced=self.config.command_policy == "projected",
                    cleanup_behavior=("no new joint target; device retains its last goal"
                                      if self.config.command_policy == "reference" else
                                      "hold measured pose or retained grasp goal"),
                    automatic_homing=False, automatic_emergency_reset=False)
