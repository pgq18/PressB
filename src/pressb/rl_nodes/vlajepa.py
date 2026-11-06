"""Compatibility alias for :mod:`pressb.models.vlajepa`."""
import sys
from pressb.models import vlajepa as _implementation
sys.modules[__name__] = _implementation
