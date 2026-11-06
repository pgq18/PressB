"""Compatibility alias for :mod:`pressb.tasks.pap.recording`."""
import sys
from pressb.tasks.pap import recording as _implementation

sys.modules[__name__] = _implementation
