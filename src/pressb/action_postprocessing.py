"""Optional causal smoothing for seven-dimensional joint commands at 30 Hz.

Commands contain six absolute arm angles in radians and the full gripper width
in metres. Any task executor using this action representation may reuse this
module. Apply once per 30 Hz control command, not once per physics tick. Keep
one instance for an entire episode, including policy chunk boundaries.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np


def _validated_command(value):
    command = np.asarray(value)
    if (command.shape != (7,) or command.dtype.kind not in "fiu"
            or not np.isfinite(command).all()):
        raise ValueError("Expected seven finite real joint/gripper values in rad/m")
    return command


class JointActionPostprocessor:
    """Smooth arm joints from the previous sent command; leave the gripper intact."""

    def __init__(self, mode="none", alpha=.8):
        if mode not in ("none", "joint_ema"):
            raise ValueError("Postprocessing mode must be none or joint_ema")
        if (isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real)
                or not math.isfinite(alpha) or not 0 < alpha <= 1):
            raise ValueError("Postprocessing alpha must be finite and in (0, 1]")
        self.mode, self.alpha = mode, float(alpha)
        self._previous = None
        self._applied = 0

    def reset(self, initial):
        """Start an episode from its settled measured joint/gripper state."""
        self._previous = _validated_command(initial).copy()
        self._applied = 0

    def apply(self, command):
        raw = _validated_command(command)
        if self.mode == "joint_ema" and self._previous is None:
            raise RuntimeError("Reset joint_ema with the initial measured state before applying actions")
        if self.mode == "none" or self.alpha == 1.:
            sent = raw.copy()
        else:
            sent = raw.astype(float, copy=True)
            sent[:6] = self.alpha * sent[:6] + (1. - self.alpha) * self._previous[:6]
        # Copy separately so caller mutation cannot change the next EMA state.
        self._previous = sent.copy()
        self._applied += 1
        return sent

    def summary(self):
        return dict(mode=self.mode, alpha=self.alpha, control_hz=30,
            formula=("sent[t][:6] = alpha * raw[t][:6] + (1-alpha) * sent[t-1][:6]"
                     if self.mode == "joint_ema" else "sent[t] = raw[t]"),
            initial_state="settled measured joint angles and full gripper width",
            reset_rule="once per episode; retain state across policy chunks",
            gripper_filtered=False, gripper_rule="sent[t][6] = raw[t][6]",
            initialized=self._previous is not None, applied_actions=self._applied)
