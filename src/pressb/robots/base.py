"""Physical robot boundary; all positions use the same SI units as simulation."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class RobotHardware(Protocol):
    """One physical arm; its measured/command coordinate ABI is declared by describe()."""

    def connect(self) -> None: ...

    def read_state(self) -> np.ndarray: ...

    def command(self, target: np.ndarray) -> np.ndarray: ...

    def hold(self) -> np.ndarray: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...

    def status(self) -> dict[str, Any]: ...

    def describe(self) -> dict[str, Any]: ...
