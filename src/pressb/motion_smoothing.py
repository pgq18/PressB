"""Causal joint-command smoothing after the policy controller's interpolation.

One instance belongs to one environment and one episode. Call ``step`` exactly
once per physics command and keep the instance across model chunks. This filter
has no access to model poses, goals or observations and does not resample time.
"""
from __future__ import annotations

from collections import deque

import numpy as np


SUPPORTED_WINDOWS = (1, 3, 5, 7, 9, 11)
DEFAULT_SMOOTHING_WINDOW = 3


def validate_smoothing_window(window):
    """Return a supported integer window; reject booleans and rounded floats."""
    if (isinstance(window, (bool, np.bool_)) or not isinstance(window, (int, np.integer))
            or window not in SUPPORTED_WINDOWS):
        raise ValueError(f"Smoothing window must be one of {SUPPORTED_WINDOWS}")
    return int(window)


def smoothing_settings(window, physics_hz=120):
    """Serializable execution semantics, including the FIR's nominal delay."""
    window = validate_smoothing_window(window)
    if (isinstance(physics_hz, (bool, np.bool_))
            or not isinstance(physics_hz, (int, np.integer)) or physics_hz <= 0):
        raise ValueError("physics_hz must be a positive integer")
    physics_hz = int(physics_hz)
    return {
        "kind": "linear_joint_interpolation_then_causal_mean",
        "window": window,
        "physics_hz": physics_hz,
        "nominal_delay_s": (window - 1) / (2 * physics_hz),
        "initial_history": "repeat_initial_command",
        "reset": "per_episode",
        "history": "cross_chunk",
    }


def _joint_copy(value):
    """Validate before changing history, and never retain caller-owned arrays."""
    try:
        raw = np.asarray(value)
        if np.iscomplexobj(raw):
            raise ValueError("Complex joint commands are not supported")
        result = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Joint command must contain six finite real values") from error
    if result.shape != (6,) or not np.isfinite(result).all():
        raise ValueError("Joint command must contain six finite real values")
    return result


class JointCommandSmoother:
    """Trailing mean of the latest ``window`` six-joint physics commands.

    Repeated initial history makes startup causal and deterministic. An input
    that respects joint limits and per-tick velocity bounds keeps those bounds
    under averaging (up to floating-point roundoff), provided the first input
    obeys the velocity bound relative to ``initial_q``. Acceleration/jerk limits
    and task success are not guaranteed by this filter.
    """

    def __init__(self, initial_q, window=DEFAULT_SMOOTHING_WINDOW):
        self.window = validate_smoothing_window(window)
        initial = _joint_copy(initial_q)
        self._history = deque((initial.copy() for _ in range(self.window)), maxlen=self.window)

    def step(self, q_linear):
        """Consume one interpolated command and return a new float64 vector."""
        command = _joint_copy(q_linear)
        self._history.append(command)
        if self.window == 1:
            # Avoid arithmetic so disabled smoothing exactly reproduces the
            # previous float64 command, including its zero sign and bit pattern.
            return command.copy()
        return np.mean(np.stack(self._history), axis=0, dtype=np.float64)
