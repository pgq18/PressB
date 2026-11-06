"""Compatibility alias for :mod:`pressb.tasks.pap.evaluation`."""
import sys
from pressb.tasks.pap import evaluation as _implementation

sys.modules[__name__] = _implementation
