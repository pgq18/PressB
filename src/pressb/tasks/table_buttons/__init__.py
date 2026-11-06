"""Physical table-button task implementation and lazy runtime plugin."""

__all__ = ["TableButtonsBackend", "create_backend"]


def __getattr__(name):
    if name in __all__:
        from . import backend
        return getattr(backend, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
