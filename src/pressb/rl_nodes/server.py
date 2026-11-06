"""Compatibility alias for :mod:`pressb.runtime.server`."""
import sys
from pressb.runtime import server as _implementation

sys.modules[__name__] = _implementation
