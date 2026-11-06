"""Compatibility alias for :mod:`pressb.tasks.pap.scene`."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("pressb.tasks.pap.scene")
