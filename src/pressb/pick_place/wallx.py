"""Compatibility alias for :mod:`pressb.compat.wallx`."""
import sys
from pressb.compat import wallx as _implementation

sys.modules[__name__] = _implementation
