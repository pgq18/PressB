"""Lazy built-in plugins. Listing configurations never imports CUDA or Isaac.

External plugins need no registry edits: node/experiment configs may use an
explicit ``factory: package.module:create_backend`` instead of a short name.
"""
from __future__ import annotations

_PLUGINS = {
    "task": {
        "pap": "pressb.tasks.pap.vector:create_backend",
        "pap_process": "pressb.tasks.pap.backend:create_backend",
        "pap_recording": "pressb.tasks.pap.recording:create_backend",
        "elevator": "pressb.tasks.elevator.native:create_backend",
        "elevator_remote": "pressb.tasks.elevator.backend:create_backend",
        "table_buttons": "pressb.tasks.table_buttons.backend:create_backend",
    },
    "model": {
        "walloss": "pressb.models.walloss:create_backend",
        "vlajepa": "pressb.models.vlajepa:create_backend",
        "vlajepa_joint": "pressb.models.vlajepa_joint:create_backend",
    },
}


def available(kind=None):
    if kind is None:
        return {key: dict(value) for key, value in _PLUGINS.items()}
    if kind not in _PLUGINS:
        raise ValueError(f"Unknown plugin kind: {kind!r}")
    return dict(_PLUGINS[kind])


def resolve(kind, name):
    choices = available(kind)
    if not isinstance(name, str) or name not in choices:
        raise ValueError(f"Unknown {kind} plugin {name!r}; available: {', '.join(choices)}")
    return choices[name]
