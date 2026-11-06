"""Compatibility alias for :mod:`pressb.tasks.pap.task`."""
import sys
from pressb.tasks.pap import task as _implementation

sys.modules[__name__] = _implementation
