"""Compatibility alias for :mod:`pressb.tasks.pap.backend`."""
import sys
from pressb.tasks.pap import backend as _implementation

sys.modules[__name__] = _implementation
