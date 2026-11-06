"""Compatibility alias for :mod:`pressb.tasks.table_buttons.scene`."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("pressb.tasks.table_buttons.scene")
