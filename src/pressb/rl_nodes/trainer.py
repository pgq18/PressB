"""Compatibility alias for :mod:`pressb.runtime.trainer`."""
import sys
from pressb.runtime import trainer as _implementation

sys.modules[__name__] = _implementation
