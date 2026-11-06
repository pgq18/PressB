"""Instruction semantics for the trained elevator floor-button task."""
from __future__ import annotations

import re

TASK_PATTERN = re.compile(r"Press (2[4-9]|3[0-5]) floor\.")


def task_floor(task):
    match = TASK_PATTERN.fullmatch(task) if isinstance(task, str) else None
    if match is None:
        raise ValueError("Task must be exactly 'Press 24 floor.' through 'Press 35 floor.'")
    return int(match.group(1))


