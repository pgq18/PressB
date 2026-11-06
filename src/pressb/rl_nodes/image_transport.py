"""Compatibility alias for :mod:`pressb.runtime.image_transport`."""
import sys
from pressb.runtime import image_transport as _implementation

sys.modules[__name__] = _implementation
