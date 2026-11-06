"""Compatibility alias for :mod:`pressb.models.walloss`."""
import sys
from pressb.models import walloss as _implementation
sys.modules[__name__] = _implementation
