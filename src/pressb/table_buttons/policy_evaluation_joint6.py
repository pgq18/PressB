"""Compatibility alias for :mod:`pressb.compat.table_buttons_policy`."""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("pressb.compat.table_buttons_policy")
