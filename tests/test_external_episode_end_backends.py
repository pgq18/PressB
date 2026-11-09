"""CPU checks for externally ending slots without another physical transition."""
from copy import deepcopy
import threading

import numpy as np
import pytest

from pressb.tasks.elevator.backend import ElevatorBackend
from pressb.tasks.elevator.fast_simulation import IndexedSimulationService
from pressb.tasks.elevator.legacy_native import NativeElevatorBackend, _LocalLegacyClient
from pressb.tasks.pap.backend import PAPParallelBackend
from test_rl_nodes_elevator import PhysicsFixture, actions as elevator_actions
from test_rl_nodes_pap import FakeWorker
from test_rl_nodes_pap_vector import fake_backend as vector_backend, actions as vector_actions
from test_table_buttons_backend import backend as table_backend, actions as table_actions, reset as table_reset


class NativePhysics(PhysicsFixture):
    """Retain the real engine's independent active-slot barrier and tick state."""

    def __init__(self):
        super().__init__()
        self._items = None
        self.recording = set()
        self.world_ticks = 0

    def reset(self, episodes, seed):
        rows = super().reset(episodes, seed)
        self._items = [dict(done=False, termination=None, ticks=0) for _ in episodes]
        self.recording = set(range(len(episodes)))
        return rows

    def step(self, actions):
        assert set(actions) == {i for i, item in enumerate(self._items) if not item["done"]}
        rows = super().step({i: np.asarray(value).tolist() for i, value in actions.items()})
        self.world_ticks += max(row["executed_physics_steps"] for row in rows)
        for row in rows:
            self._items[row["env_id"]]["ticks"] += row["executed_physics_steps"]
        return rows

    def reset_envs(self, episodes, seed):
        assert all(self._items[i]["done"] for i in episodes)
        rows = super().reset_envs(episodes, seed)
        for i in episodes:
            self._items[i] = dict(done=False, termination=None, ticks=0)
            self.recording.add(i)
        return rows


def native_backend(*, initialized=True):
    node = object.__new__(NativeElevatorBackend)
    node._owner = threading.get_ident()
    node._engine = NativePhysics()
    node._legacy_service = IndexedSimulationService(node._engine)
    ElevatorBackend.__init__(node, {"endpoint": "http://unused"},
                             client=_LocalLegacyClient(node._legacy_service))
    if initialized:
        node.reset([0, 1], 7)
        node.step(elevator_actions())
    return node


def process_backend(*, initialized=True):
    node = object.__new__(PAPParallelBackend)
    node.config = dict(horizon=2, command_timeout=1.)
    node._broken = node._closed = False
    node._ready = [initialized, initialized]
    node.lower, node.upper = np.full(7, -1.), np.ones(7)
    node.events = []
    node.workers = [FakeWorker(i, node.events) for i in range(2)]
    return node


def make_backend(kind, *, initialized=True):
    if kind == "process":
        return process_backend(initialized=initialized)
    if kind == "vector":
        node = vector_backend()
        node.ready[:] = initialized
        node.outcomes[0] = node.outcomes[1]
        return node
    if kind == "table":
        node = table_backend(max_episode_seconds=1.)
        if initialized:
            table_reset(node)
        return node
    return native_backend(initialized=initialized)


def snapshot(node, kind):
    if kind == "process":
        return deepcopy((node._ready, node.events, [worker.clock for worker in node.workers]))
    if kind == "vector":
        return node.ready.tolist(), node.ticks.tolist(), node.world.count
    if kind == "table":
        return node._ready, None if node.environment is None else node.environment.total_ticks
    return deepcopy((node._items, node._legacy_service.items, node._engine._items,
                     node._engine.recording, node._step, node._legacy_service.step_id,
                     node._engine.calls, node._engine.world_ticks))


@pytest.mark.parametrize("kind", ["process", "vector", "table", "native"])
@pytest.mark.parametrize("ids", [None, (), [], [0, 0], [True], [False], [0, True],
                                [0.], ["0"], [None], [[0]], [-1], [2], [0, 2]])
def test_invalid_ids_rejected_before_any_slot_changes(kind, ids):
    node = make_backend(kind)
    before = snapshot(node, kind)
    with pytest.raises(ValueError):
        node.end_episode(ids)
    assert snapshot(node, kind) == before


@pytest.mark.parametrize("kind", ["process", "vector", "table", "native"])
def test_uninitialized_and_already_ended_slots_are_rejected(kind):
    node = make_backend(kind, initialized=False)
    before = snapshot(node, kind)
    with pytest.raises(ValueError, match="Only active" if kind == "vector" else "initialized active"):
        node.end_episode([0])
    assert snapshot(node, kind) == before
    node = make_backend(kind)
    node.end_episode([0])
    before = snapshot(node, kind)
    with pytest.raises(ValueError, match="Only active" if kind == "vector" else "initialized active"):
        node.end_episode([0])
    assert snapshot(node, kind) == before


@pytest.mark.parametrize("kind", ["process", "vector", "native"])
def test_active_subset_is_fully_validated_before_mutating(kind):
    node = make_backend(kind)
    node.end_episode([1])
    before = snapshot(node, kind)
    with pytest.raises(ValueError, match="Only active" if kind == "vector" else "initialized active"):
        node.end_episode([0, 1])
    assert snapshot(node, kind) == before


def test_process_end_sends_no_worker_commands_and_allows_peer_step_and_indexed_reset(monkeypatch):
    node = process_backend()
    node.step(vector_actions(prefix=1))
    before_calls, clocks = deepcopy(node.events), [worker.clock for worker in node.workers]
    node.end_episode([0])
    assert node.events == before_calls
    assert [worker.clock for worker in node.workers] == clocks
    assert node._ready == [False, True]
    with pytest.raises(ValueError, match="requires reset"):
        node.step(vector_actions(ids=(0,), prefix=1))
    node.step(vector_actions(ids=(1,), prefix=1))
    assert [worker.clock for worker in node.workers] == [1, 2]
    monkeypatch.setattr("pressb.tasks.pap.backend.sample_reset", lambda *args: {"seed": args[1]})
    node.reset([0], 8)
    assert node._ready == [True, True]
    assert [worker.clock for worker in node.workers] == [0, 2]


def test_vector_end_preserves_ticks_and_observation_and_removes_slot_from_barrier():
    node = make_backend("vector")
    rows = node.step(vector_actions(prefix=1))
    original = deepcopy(rows)
    counts = node.contact_counts.copy()
    node.end_episode([0])
    assert node.world.count == 1 and node.ticks.tolist() == [1, 1]
    np.testing.assert_array_equal(node.contact_counts, counts)
    assert rows == original
    with pytest.raises(ValueError, match="every active"):
        node.step(vector_actions(prefix=1))
    peer, = node.step(vector_actions(ids=(1,), prefix=1))
    assert peer["env_id"] == 1 and peer["reward"] == 0.
    assert node.world.count == 2 and node.ticks.tolist() == [1, 2]
    assert not node.ready[0] and node.ready[1]


def test_table_end_preserves_completed_transition_and_reset_reactivates():
    node = make_backend("table")
    row, = node.step([dict(env_id=0, actions=table_actions(node, 1))])
    before = deepcopy(row)
    clock, calls = node.environment.total_ticks, len(node.environment.sent)
    node.end_episode([0])
    assert node.environment.total_ticks == clock and len(node.environment.sent) == calls
    assert row == before and not row["info"]["success"] and row["reward"] == 0.
    with pytest.raises(ValueError, match="Reset"):
        node.step([dict(env_id=0, actions=table_actions(node, 1))])
    table_reset(node)
    node.step([dict(env_id=0, actions=table_actions(node, 1))])
    assert node.environment.total_ticks > clock


def test_native_end_synchronizes_three_barriers_and_preserves_peer_and_cached_transition():
    node = native_backend()
    service, engine = node._legacy_service, node._engine
    bridge_before, legacy_before, engine_before = deepcopy((node._items, service.items, engine._items))
    cached = deepcopy(service._responses)
    last_step = node._step
    calls, ticks = deepcopy(engine.calls), engine.world_ticks
    node.end_episode([0])
    assert node._step == service.step_id == last_step
    assert engine.calls == calls and engine.world_ticks == ticks
    assert engine.recording == {1} and service._active() == [1]
    assert node._items[1] == bridge_before[1]
    assert service.items[1] == legacy_before[1] and engine._items[1] == engine_before[1]
    assert service._responses == cached
    for current, prior in ((node._items[0], bridge_before[0]), (service.items[0], legacy_before[0])):
        assert current["terminated"] is True and current["truncated"] is False
        assert current["info"]["termination"] == "external_episode_end"
        for key in ("observation", "episode_id", "reward", "discount", "executed_physics_steps"):
            assert current[key] == prior[key]
        assert current["info"]["success"] is False
    assert engine._items[0] == {**engine_before[0], "done": True, "termination": "external_episode_end"}

    peer, = node.step(elevator_actions(ids=(1,)))
    assert peer["env_id"] == 1 and peer["reward"] == 0.
    assert engine._items[0]["ticks"] == engine_before[0]["ticks"]
    assert node._items[0]["observation"] == bridge_before[0]["observation"]
    assert node._items[0]["executed_physics_steps"] == 0
    peer_bridge, peer_engine = deepcopy((node._items[1], engine._items[1]))
    reset_clock = engine.world_ticks
    fresh, = node.reset([0], 8)
    assert fresh["env_id"] == 0
    assert node._items[0]["episode_id"] != bridge_before[0]["episode_id"]
    assert node._items[1] == peer_bridge and engine._items[1] == peer_engine
    assert engine.world_ticks == reset_clock and engine.recording == {0, 1}
    node.step(elevator_actions())


@pytest.mark.parametrize("layer", ["bridge", "service", "engine"])
def test_native_checks_each_layer_before_any_mutation(layer):
    node = native_backend()
    if layer == "bridge":
        del node._items[1]
    elif layer == "service":
        node._legacy_service.items[1]["truncated"] = True
    else:
        node._engine._items[1]["done"] = True
    before = snapshot(node, "native")
    with pytest.raises(ValueError, match="initialized active"):
        node.end_episode([0, 1])
    assert snapshot(node, "native") == before


def test_native_rejects_other_threads_before_changing_any_state():
    node = native_backend()
    before, errors = snapshot(node, "native"), []
    def attempt():
        try:
            node.end_episode([0])
        except RuntimeError as error:
            errors.append(str(error))
    thread = threading.Thread(target=attempt)
    thread.start()
    thread.join()
    assert errors and "owning simulator thread" in errors[0]
    assert snapshot(node, "native") == before


def test_remote_elevator_does_not_advertise_an_unimplemented_operation():
    assert not callable(getattr(ElevatorBackend, "end_episode", None))
