"""Compatibility alias for :mod:`pressb.runtime.client`."""
import sys
from pressb.runtime import client as _implementation

sys.modules[__name__] = _implementation
