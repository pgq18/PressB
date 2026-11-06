"""Compatibility alias for :mod:`pressb.tasks.elevator.backend`."""
import sys
from pressb.tasks.elevator import backend as _implementation

sys.modules[__name__] = _implementation
