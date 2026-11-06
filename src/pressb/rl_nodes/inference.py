"""Compatibility alias for :mod:`pressb.runtime.inference`."""
import sys
from pressb.runtime import inference as _implementation

sys.modules[__name__] = _implementation
