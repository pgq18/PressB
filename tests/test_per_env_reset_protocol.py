"""CPU checks for batched independent reset seeds; Isaac smoke is separate."""
from copy import deepcopy

import pytest

from pressb.runtime.contract import normalize_env_seeds
from pressb.runtime.simulation import SimulationService
from pressb.tasks.pap.backend import PAPParallelBackend, sample_reset
from pressb.tasks.pap.vector import PAPVectorBackend
from pressb.tasks.pap.parallel import PAPTask


def metadata():
    return dict(schema_version=2, node_type="simulation", task_id="fixture", num_envs=3,
        observation={"state": {"names": ["joint"], "units": ["rad"]}, "cameras": []},
        action={"representation": "joint_position", "frame": "robot_base", "names": ["joint"],
                "units": ["rad"], "horizon": 8, "control_hz": 30},
        control_state_dim=0, task_features_dim=0, reset_mode="indexed",
        discount={"gamma": .99, "timebase_hz": 30})


class LegacyBackend:
    def __init__(self):
        self.calls = []

    def describe(self):
        return metadata()

    def reset(self, env_ids, seed, options):
        self.calls.append((env_ids.copy(), seed, deepcopy(options)))
        return [dict(env_id=i, observation={"state": [0.], "images": {}, "task": "Place it."},
                     info={"seed": seed}) for i in env_ids]


class SeedBackend(LegacyBackend):
    def describe(self):
        return {**metadata(), "capabilities": ["per_env_reset_seeds"]}

    def reset(self, env_ids, seed, options, *, env_seeds=None):
        rows = super().reset(env_ids, seed, options)
        self.calls[-1] += (deepcopy(env_seeds),)
        if env_seeds is not None:
            for row in rows:
                row["info"]["seed"] = env_seeds[row["env_id"]]
        return rows


def request(**updates):
    return dict(run_id="run", request_id="r0", env_ids=[2, 0], seed=17, options={}, **updates)


def test_normalization_preserves_per_slot_values_and_does_not_mutate_input():
    supplied = {"2": 2003, 0: 41}
    assert normalize_env_seeds([2, 0], supplied) == {2: 2003, 0: 41}
    assert supplied == {"2": 2003, 0: 41}
    assert normalize_env_seeds([2, 0], None) is None


INVALID_MAPPINGS = [
    [], "1", 3, True, {}, {"0": 1}, {"0": 1, "2": 2, "1": 3},
    {"0": 1, "2": 2, 2: 3}, {False: 1, "2": 2}, {0.: 1, "2": 2},
    {"00": 1, "2": 2}, {"+0": 1, "2": 2}, {" 0": 1, "2": 2},
    {"０": 1, "2": 2}, {"0": 1, "-2": 2}, {0: 1, -2: 2},
    {"0": 1, "2": True}, {"0": 1, "2": 2.}, {"0": 1, "2": -1},
    {"0": 1, "2": "3"},
]


@pytest.mark.parametrize("mapping", INVALID_MAPPINGS + [None])
def test_invalid_seed_maps_are_rejected_before_mutation_or_fault(mapping):
    backend = SeedBackend()
    service = SimulationService(backend)
    with pytest.raises(ValueError):
        service.reset(request(env_seeds=mapping))
    assert backend.calls == []
    assert service.fault is None and service.run_id is None
    assert not service.slots and not service.requests and not service.seen_requests
    # A corrected retry may reuse the same uncommitted request ID.
    assert len(service.reset(request(env_seeds={"0": 1, "2": 2}))["items"]) == 2


def test_unsupported_seed_map_rejected_without_breaking_legacy_reset():
    backend = LegacyBackend()
    service = SimulationService(backend)
    with pytest.raises(ValueError, match="per_env_reset_seeds"):
        service.reset(request(env_seeds={"0": 1, "2": 2}))
    assert backend.calls == [] and service.fault is None
    rows = service.reset(request())["items"]
    assert backend.calls == [([2, 0], 17, {})]
    assert [row["info"]["seed"] for row in rows] == [17, 17]


def test_batch_seed_mapping_is_idempotent_and_fingerprinted():
    backend = SeedBackend()
    service = SimulationService(backend)
    first = service.reset(request(env_seeds={2: 27, 0: 50}))
    assert [row["env_id"] for row in first["items"]] == [2, 0]
    assert [row["info"]["seed"] for row in first["items"]] == [27, 50]
    assert backend.calls == [([2, 0], 17, {}, {2: 27, 0: 50})]
    # A JSON transport stringifies integer object keys. The same request must
    # still return the identical episode ownership, without a second reset.
    assert service.reset(request(env_seeds={"0": 50, "2": 27})) == first
    with pytest.raises(ValueError, match="different operation or content"):
        service.reset(request(env_seeds={"2": 28, "0": 50}))
    assert len(backend.calls) == 1 and service.fault is None


@pytest.mark.parametrize("seed", [None, True, -1, 1.5])
def test_shared_seed_is_still_required_with_per_env_mapping(seed):
    backend = SeedBackend()
    service = SimulationService(backend)
    payload = request(env_seeds={"0": 1, "2": 2})
    payload["seed"] = seed
    with pytest.raises(ValueError, match="Reset seed"):
        service.reset(payload)
    assert backend.calls == [] and service.fault is None


def test_new_seed_api_preserves_initial_all_envs_and_active_episode_guards():
    backend = SeedBackend()
    backend.describe = lambda: {**metadata(), "initial_reset": "all_envs",
                                "capabilities": ["per_env_reset_seeds"]}
    service = SimulationService(backend)
    with pytest.raises(ValueError, match="all environments"):
        service.reset(request(env_seeds={"0": 1, "2": 2}))
    initial = request(env_seeds={"0": 1, "1": 3, "2": 2})
    initial["env_ids"] = [2, 0, 1]
    service.reset(initial)
    active = request(env_seeds={"0": 1, "2": 2})
    active["request_id"] = "r1"
    with pytest.raises(ValueError, match="active episode"):
        service.reset(active)
    assert len(backend.calls) == 1 and service.fault is None


def sampling_config():
    return dict(num_envs=3, command_timeout=2., task="Place it.", model_inputs={"source": "fixture"},
        objects=[dict(center_xy=[.3, 0.], yaw_deg=0., color=[1., 0., 0.],
                      geometry={"shape": "cube", "size_xyz": [.029]*3})],
        randomization={"position": "polygon", "center_polygon": [[.2, -.1], [.4, -.1], [.4, .1], [.2, .1]],
                       "colors": ["red", "blue", "yellow"], "yaw_range_deg": [-45., 45.]})


def vector():
    backend = object.__new__(PAPVectorBackend)
    backend.config = sampling_config()
    backend.num_envs = 3
    backend.task = object.__new__(PAPTask)
    backend.task.config = backend.config
    backend.reset_specs = [{"task": "already initialized"} for _ in range(3)]
    backend._check = lambda: None
    backend.mutations = []
    def reset_once(ids, specs):
        backend.mutations.append((ids.copy(), deepcopy(specs)))
        return specs
    backend._reset = reset_once
    return backend


def test_pap_vector_batch_matches_old_singleton_sampling_exactly():
    backend = vector()
    ids, seeds = [2, 0], {"0": 104003, "2": 3011}
    options = {"task": "Global instruction", "by_env": {"0": {"color": "blue"},
               2: {"yaw_deg": 7., "model_inputs": {"source": "override"}}}}
    original_options = deepcopy(options)
    expected = {i: backend._prepare_reset([i], seeds[str(i)], options)[i] for i in ids}
    actual = backend.reset(ids, 999999, options, env_seeds=seeds)
    assert actual == expected
    assert len(backend.mutations) == 1 and backend.mutations[0][0] == ids
    assert options == original_options
    # Grouping and the shared fallback seed do not perturb either slot's RNG.
    assert backend._prepare_reset([0, 2], 0, options, env_seeds=seeds) == expected


def test_pap_vector_default_seed_keeps_old_sampling():
    backend = vector()
    ids = [2, 0]
    result = backend.reset(ids, 91)
    assert result == {i: sample_reset(backend.config, 91, i, {}) for i in ids}
    assert len(backend.mutations) == 1


@pytest.mark.parametrize("mapping", [{"0": 4}, {"0": 1, "2": False}, {"0": 1, 0: 2, "2": 3}])
def test_pap_vector_bad_seed_map_does_not_enter_reset(mapping):
    backend = vector()
    with pytest.raises(ValueError):
        backend.reset([2, 0], 91, env_seeds=mapping)
    assert backend.mutations == []


def test_pap_vector_all_options_validate_before_world_mutation():
    backend = vector()
    with pytest.raises(ValueError, match="Unknown color"):
        backend.reset([2, 0], 91, {"by_env": {"0": {"color": "invalid"}}}, env_seeds={2: 4, 0: 5})
    assert backend.mutations == []


class Worker:
    def __init__(self, env_id, events):
        self.env_id, self.events = env_id, events

    def send(self, method, payload):
        self.events.append(("send", self.env_id, method, deepcopy(payload)))
        self.payload = deepcopy(payload)

    def receive(self, timeout):
        self.events.append(("receive", self.env_id))
        return {"env_id": self.env_id, "info": {"reset": deepcopy(self.payload)}}


def parallel():
    backend = object.__new__(PAPParallelBackend)
    backend.config = sampling_config()
    backend._broken = backend._closed = False
    backend._ready = [False]*3
    backend.events = []
    backend.workers = [Worker(i, backend.events) for i in range(3)]
    return backend


def test_pap_process_batch_sends_all_before_receiving_and_keeps_slot_seed():
    backend = parallel()
    options = {"by_env": {"0": {"color": "blue"}}}
    rows = backend.reset([2, 0], 999, options, env_seeds={2: 71, "0": 23})
    assert [event[:2] for event in backend.events] == [
        ("send", 2), ("send", 0), ("receive", 2), ("receive", 0)]
    assert rows[0]["info"]["reset"] == sample_reset(backend.config, 71, 2, {})
    assert rows[1]["info"]["reset"] == sample_reset(backend.config, 23, 0, {"color": "blue"})
    assert backend._ready == [True, False, True]


def test_pap_process_bad_last_seed_or_options_send_no_worker_commands():
    backend = parallel()
    with pytest.raises(ValueError):
        backend.reset([2, 0], 9, env_seeds={2: 1, 0: -1})
    with pytest.raises(ValueError):
        backend.reset([2, 0], 9, {"by_env": {"0": {"color": "invalid"}}}, env_seeds={2: 1, 0: 2})
    assert backend.events == []
    assert backend._ready == [False]*3 and not backend._broken
