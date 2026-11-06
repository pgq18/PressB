"""Compatibility alias for :mod:`pressb.runtime.contract`."""
import sys
from pressb.runtime import contract as _implementation

sys.modules[__name__] = _implementation
