"""Structural task interface; implementations own physics and task semantics.

The runtime owns RPC identity, validation, ordering, replay, and scheduling.
Backends report measured state, actual duration, and physical outcomes. Optional
capabilities (e.g. fused resets) must be advertised by describe(), never inferred.
"""
from typing import Protocol, runtime_checkable


@runtime_checkable
class TaskPlugin(Protocol):
    def describe(self) -> dict: ...
    def reset(self, env_ids: list[int], seed: int, options: dict) -> list[dict]: ...
    def step(self, commands: list[dict]) -> list[dict]: ...
    def close(self) -> None: ...
