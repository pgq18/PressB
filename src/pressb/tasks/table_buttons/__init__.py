"""Physical table-button task plugin for the common simulation service."""

from .backend import TableButtonsBackend, create_backend

__all__ = ["TableButtonsBackend", "create_backend"]
