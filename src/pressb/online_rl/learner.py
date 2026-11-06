"""Compatibility alias for :mod:`pressb.runtime.learner`."""
import sys
from pressb.runtime import learner as _implementation

sys.modules[__name__] = _implementation
