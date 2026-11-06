"""Units and validation for historical seven-channel table-button policies."""
import numpy as np


SCALE = np.asarray([180.] * 6 + [8.])
POLICY_TO_SIM = np.asarray([np.pi / 180] * 6 + [.01])


class InvalidPolicyAction(ValueError):
    """A completed response contains unusable model actions (policy failure)."""


def joint7(value):
    result = np.asarray(value)
    if result.dtype.kind not in "fiu" or result.shape != (7,) or not np.isfinite(result).all():
        raise ValueError("Expected seven finite real joint/gripper values")
    return result.astype(np.float64, copy=True)


def to_policy_units(rad_m):
    return joint7(rad_m) / POLICY_TO_SIM


def to_sim_units(deg_cm):
    return joint7(deg_cm) * POLICY_TO_SIM
