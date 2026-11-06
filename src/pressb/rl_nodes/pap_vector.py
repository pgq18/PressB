"""Compatibility alias for :mod:`pressb.tasks.pap.vector`."""
import sys
from pressb.tasks.pap import vector as _implementation

sys.modules[__name__] = _implementation
