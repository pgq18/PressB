"""Compatibility alias for :mod:`pressb.runtime.simulation`."""
import sys
from pressb.runtime import simulation as _implementation

sys.modules[__name__] = _implementation
