"""Compatibility alias for :mod:`pressb.runtime.rpc`."""
import sys
from pressb.runtime import rpc as _implementation

sys.modules[__name__] = _implementation
