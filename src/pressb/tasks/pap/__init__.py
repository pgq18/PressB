"""PAP scene, task rules, sampling, initialization, and lazy simulation backends.

Pure NumPy task helpers are importable without Isaac Sim. Backend classes stay
lazy so dataset utilities and historical import aliases need no simulator.
"""
from importlib import import_module

__all__ = ["PAPParallelBackend", "PAPVectorBackend", "PAPRecordingVectorBackend"]
_MODULES = {"PAPParallelBackend": ".backend", "PAPVectorBackend": ".vector",
            "PAPRecordingVectorBackend": ".recording"}


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_MODULES[name], __name__), name)
    globals()[name] = value
    return value
