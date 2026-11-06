from copy import deepcopy

import numpy as np
import pytest

from pressb.runtime.contract import negotiate, validate_metadata
from pressb.runtime.simulation import SimulationService


def metadata(kind="simulation"):
    value = dict(schema_version=2, node_type=kind,
                 observation={"state": {"names": ["joint1", "gripper"], "units": ["rad", "m"]}, "cameras": []},
                 action={"representation": "joint_position", "frame": "robot_base", "names": ["joint1", "gripper"],
                         "units": ["rad", "m"], "horizon": 32, "control_hz": 30})
    if kind == "inference":
        value.update(model_id="fixture", feature_dim=4, noise={"shape": [32, 20]},
                     capabilities=["encode", "decode", "features", "context_cache", "external_initial_noise"])
    else:
        value.update(task_id="fixture", num_envs=2, control_state_dim=2, task_features_dim=0,
                     discount={"gamma": .99, "timebase_hz": 30}, reset_mode="indexed")
    return value


class Backend:
    def __init__(self):
        self.calls = []
        self.count = [0, 0]
        self.fail = False

    def describe(self):
        return metadata()

    def observation(self, i):
        return dict(state=[float(self.count[i]), .01], images={}, task="Place it.",
                    control_state=[0., .01], task_features=[])

    def reset(self, env_ids, seed, options):
        self.calls.append(("reset", env_ids.copy()))
        for i in env_ids:
            self.count[i] = 0
        return [dict(env_id=i, observation=self.observation(i)) for i in env_ids]

    def step(self, items):
        self.calls.append(("step", [x["env_id"] for x in items]))
        if self.fail:
            raise RuntimeError("lost render after physics")
        result = []
        for item in items:
            i = item["env_id"]
            self.count[i] += 1
            done = i == 0 or self.count[i] >= 3
            result.append(dict(env_id=i, observation=self.observation(i), reward=float(done),
                               discount=.99 ** .5, terminated=done, truncated=False,
                               info={"success": done, "elapsed_control_steps": .5, "physics_steps": 2}))
        return result


def reset(service, ids=(0, 1), rid="r0"):
    return service.reset(dict(run_id="run", request_id=rid, env_ids=list(ids), seed=5, options={}))


def step_payload(reset_result, rid="s0"):
    return dict(run_id="run", request_id=rid, items=[
        {k: row[k] for k in ("env_id", "episode_id", "step_id")} | {"actions": [[.1, .02]]}
        for row in reset_result["items"]])


def test_independent_terminal_reset_and_exactly_once():
    backend = Backend()
    service = SimulationService(backend)
    initial = reset(service)
    request = step_payload(initial)
    response = service.step(request)
    assert response["items"][0]["terminated"] and not response["items"][1]["terminated"]
    assert response["items"][0]["observation"]["state"][0] == 1
    assert service.step(deepcopy(request)) == response
    assert backend.count == [1, 1]
    new = reset(service, (0,), "r1")
    assert backend.count == [0, 1]
    assert new["items"][0]["episode_id"] != initial["items"][0]["episode_id"]
    assert service.slots[1]["episode_id"] == initial["items"][1]["episode_id"]


def test_full_request_validated_before_any_environment_advances():
    backend = Backend()
    service = SimulationService(backend)
    initial = reset(service)
    request = step_payload(initial)
    request["items"][1]["actions"] = [[float("nan"), 0.]]
    with pytest.raises(ValueError):
        service.step(request)
    assert backend.count == [0, 0] and service.fault is None
    request["items"][1]["actions"] = [[0., 0.]]
    request["items"][1]["step_id"] = 1
    with pytest.raises(ValueError, match="Stale"):
        service.step(request)
    assert backend.count == [0, 0]


def test_expired_request_cannot_repeat_physics():
    backend = Backend()
    service = SimulationService(backend, request_cache_size=1)
    initial = reset(service)
    service.step(step_payload(initial))
    with pytest.raises(ValueError, match="expired"):
        reset(service)
    assert backend.count == [1, 1]


def test_unequal_prefix_rejected_before_shared_physics_without_faulting_node():
    backend = Backend()
    backend.describe = lambda: {**metadata(), "equal_execution_horizon": True}
    service = SimulationService(backend)
    initial = reset(service)
    request = step_payload(initial)
    request["items"][1]["actions"] *= 2
    with pytest.raises(ValueError, match="equal action prefix lengths"):
        service.step(request)
    assert backend.calls == [("reset", [0, 1])]
    assert backend.count == [0, 0] and service.fault is None
    assert all(slot["step_id"] == 0 for slot in service.slots.values())
    # A corrected request can use the same uncommitted request ID and run.
    request["items"][0]["actions"] *= 2
    assert len(service.step(request)["items"]) == 2
    assert backend.count == [1, 1] and service.fault is None


@pytest.mark.parametrize("required", [None, False])
def test_independent_backend_still_accepts_different_action_prefixes(required):
    backend = Backend()
    spec = metadata()
    if required is not None:
        spec["equal_execution_horizon"] = required
    backend.describe = lambda: spec
    service = SimulationService(backend)
    request = step_payload(reset(service))
    request["items"][1]["actions"] *= 2
    assert len(service.step(request)["items"]) == 2
    assert backend.count == [1, 1]


def test_physics_failure_poisoned_not_reported_as_failed_episode():
    backend = Backend()
    service = SimulationService(backend)
    initial = reset(service)
    backend.fail = True
    with pytest.raises(RuntimeError, match="lost render"):
        service.step(step_payload(initial))
    before = len(backend.calls)
    with pytest.raises(RuntimeError, match="unknown"):
        reset(service, (0,), "r2")
    assert len(backend.calls) == before


def test_owner_and_live_episode_reset_guards():
    service = SimulationService(Backend())
    reset(service)
    with pytest.raises(ValueError, match="active episode"):
        reset(service, (1,), "other")
    with pytest.raises(ValueError, match="another run"):
        service.reset(dict(run_id="intruder", request_id="x", env_ids=[0], seed=1))
    with pytest.raises(ValueError, match="owning"):
        service.close_run({"run_id": "intruder"})
    service.close_run({"run_id": "run"})
    assert service.run_id is None


def test_actual_discount_error_is_infrastructure_failure():
    backend = Backend()
    original = backend.step
    def wrong(items):
        rows = original(items)
        rows[0]["discount"] = .9
        return rows
    backend.step = wrong
    service = SimulationService(backend)
    initial = reset(service)
    with pytest.raises(RuntimeError, match="discount"):
        service.step(step_payload(initial))
    assert service.fault is not None


def test_noise_width_independent_of_actuator_width_and_parallelism():
    sim, inference = metadata(), metadata("inference")
    result = negotiate(inference, sim, "dsrl", 8)
    assert result["noise_shape"] == [32, 20]
    assert result["action_dim"] == 2 and result["obs_dim"] == 8
    sim["num_envs"] = 9
    assert result["fingerprint"] == negotiate(inference, sim, "dsrl", 8)["fingerprint"]


@pytest.mark.parametrize("key,value", [("units", ["deg", "m"]), ("frame", "world"),
                                        ("names", ["gripper", "joint1"]), ("control_hz", 20)])
def test_incompatible_nodes_fail_before_rollout(key, value):
    sim, inference = metadata(), metadata("inference")
    sim["action"][key] = value
    with pytest.raises(ValueError, match="mismatch"):
        negotiate(inference, sim, "dsrl")


def test_missing_noise_capability_rejected():
    inference = metadata("inference")
    inference["capabilities"].remove("external_initial_noise")
    with pytest.raises(ValueError, match="capabilities"):
        negotiate(inference, metadata(), "dsrl")


def test_declared_horizon_limit():
    sim = metadata()
    sim["execution_horizons"] = [32]
    with pytest.raises(ValueError, match="horizon"):
        negotiate(metadata("inference"), sim, "dsrl", 8)


@pytest.mark.parametrize("field,value", [("num_envs", True), ("control_state_dim", -1),
                                          ("reset_mode", "cohort"), ("schema_version", 1),
                                          ("equal_execution_horizon", "true")])
def test_invalid_metadata(field, value):
    spec = metadata()
    spec[field] = value
    with pytest.raises(ValueError):
        validate_metadata(spec)


class StateOnlyBackend(Backend):
    def describe(self):
        spec = metadata()
        spec.update(capabilities=["terminal_state_only", "step_and_reset"],
                    terminal_observation_modes=["full", "state_only_if_no_bootstrap"])
        spec["observation"]["cameras"] = ["up"]
        return spec

    def observation(self, i):
        obs = super().observation(i)
        obs["images"] = {"up": {"encoding": "png_base64", "data": "fixture"}}
        return obs

    def step(self, items, *, terminal_observation_mode="full"):
        result = super().step(items)
        for row in result:
            if terminal_observation_mode != "full" and (row["terminated"] or row["truncated"]):
                row["terminal_observation_omitted"] = True
                row["observation"]["images"] = {}
        return result

    def step_and_reset(self, items, *, reset_seed, reset_options, max_active_envs):
        result = self.step(items, terminal_observation_mode="state_only_if_no_bootstrap")
        done = [row["env_id"] for row in result if row["terminated"] or row["truncated"]]
        ids = done[:max(0, max_active_envs - (len(items) - len(done)))]
        resets = self.reset(ids, reset_seed, reset_options) if ids else []
        return {"items": result, "reset_items": resets}


def test_terminal_mode_negotiation_and_default_contract_compatibility():
    inference = metadata("inference")
    backend = StateOnlyBackend()
    full = negotiate(inference, backend.describe(), "dsrl")
    assert "terminal_observation_mode" not in full
    fast = negotiate(inference, backend.describe(), "dsrl",
                     terminal_observation_mode="state_only_if_no_bootstrap")
    assert fast["terminal_observation_mode"] == "state_only_if_no_bootstrap"
    with pytest.raises(ValueError, match="terminal_state_only"):
        negotiate(inference, metadata(), "dsrl", terminal_observation_mode="state_only_if_no_bootstrap")


def test_service_keeps_terminal_state_reward_discount_but_omits_only_terminal_images():
    backend = StateOnlyBackend()
    service = SimulationService(backend)
    request = step_payload(reset(service))
    request["terminal_observation_mode"] = "state_only_if_no_bootstrap"
    response = service.step(request)
    terminal, active = response["items"]
    assert terminal["terminal_observation_omitted"] is True
    assert terminal["observation"]["images"] == {}
    assert terminal["observation"]["state"] == [1., .01]
    assert terminal["reward"] == 1.
    assert terminal["discount"] == pytest.approx(.99 ** .5)
    assert terminal["info"]["elapsed_control_steps"] == .5
    assert "terminal_observation_omitted" not in active and "up" in active["observation"]["images"]
    assert service.step(deepcopy(request)) == response  # Retry cache does not advance physics.
    assert backend.count == [1, 1]


@pytest.mark.parametrize("mode", ["unknown", "state_only_if_no_bootstrap"])
def test_unsupported_terminal_mode_rejected_before_physics(mode):
    backend = Backend()
    service = SimulationService(backend)
    request = step_payload(reset(service))
    request["terminal_observation_mode"] = mode
    with pytest.raises(ValueError, match="terminal"):
        service.step(request)
    assert backend.count == [0, 0] and service.fault is None


@pytest.mark.parametrize("invalid", ["active_missing", "terminal_unmarked", "unexpected_in_full"])
def test_service_rejects_backend_violation_of_terminal_mode(invalid):
    backend = StateOnlyBackend()
    original = backend.step
    def wrong(items, **kwargs):
        result = original(items, **kwargs)
        if invalid == "terminal_unmarked":
            result[0].pop("terminal_observation_omitted")
        elif invalid == "active_missing":
            result[1]["terminal_observation_omitted"] = True
            result[1]["observation"]["images"] = {}
        else:
            result[0]["terminal_observation_omitted"] = True
            result[0]["observation"]["images"] = {}
        return result
    backend.step = wrong
    service = SimulationService(backend)
    request = step_payload(reset(service))
    if invalid != "unexpected_in_full":
        request["terminal_observation_mode"] = "state_only_if_no_bootstrap"
    with pytest.raises(RuntimeError, match="terminal_observation_omitted"):
        service.step(request)
    assert service.fault is not None


def test_binary_png_and_rgb_validation_preserve_pixels():
    import base64
    from io import BytesIO
    from PIL import Image
    from pressb.runtime.contract import decode_image, encode_image, validate_observation
    pixels = np.arange(8 * 7 * 3, dtype=np.uint8).reshape(8, 7, 3)
    legacy = encode_image(pixels)
    binary = {"encoding": "png_bytes", "data": base64.b64decode(legacy["data"])}
    assert np.array_equal(np.asarray(decode_image(binary, require_rgb=True)), pixels)
    backend = StateOnlyBackend()
    obs = backend.observation(0)
    obs["images"]["up"] = binary
    validate_observation(obs, backend.describe()["observation"])
    stream = BytesIO()
    Image.new("L", (7, 8)).save(stream, format="PNG")
    gray = {"encoding": "png_bytes", "data": stream.getvalue()}
    assert decode_image(gray).mode == "RGB"  # Legacy conversion remains valid.
    with pytest.raises(ValueError, match="RGB"):
        decode_image(gray, require_rgb=True)
    with pytest.raises(ValueError, match="bytes"):
        decode_image({"encoding": "png_bytes", "data": "not bytes"})


def fused_payload(initial, maximum=2):
    return {**step_payload(initial), "terminal_observation_mode": "state_only_if_no_bootstrap",
            "reset_seed": 17, "reset_options": {}, "max_active_envs": maximum}


def test_fused_service_keeps_old_terminal_and_new_initial_identity_and_is_idempotent():
    backend = StateOnlyBackend()
    service = SimulationService(backend)
    initial = reset(service)
    request = fused_payload(initial)
    result = service.step_and_reset(request)
    terminal, live = result["items"]
    fresh, = result["reset_items"]
    assert terminal["episode_id"] == initial["items"][0]["episode_id"]
    assert terminal["observation"]["state"][0] == 1.
    assert terminal["terminal_observation_omitted"] and terminal["step_id"] == 1
    assert fresh["episode_id"] != terminal["episode_id"] and fresh["step_id"] == 0
    assert fresh["observation"]["state"][0] == 0.
    assert "up" in fresh["observation"]["images"]
    assert service.slots[0]["episode_id"] == fresh["episode_id"]
    assert service.slots[1]["episode_id"] == live["episode_id"]
    assert not any(slot["done"] for slot in service.slots.values())
    before = deepcopy(backend.calls)
    assert service.step_and_reset(deepcopy(request)) == result
    assert backend.calls == before
    stale = fused_payload(initial)
    stale["request_id"] = "stale"
    with pytest.raises(ValueError, match="Stale"):
        service.step_and_reset(stale)
    assert backend.calls == before


@pytest.mark.parametrize("maximum,expected", [(0, []), (1, []), (2, [0])])
def test_fused_service_budget_limits_indexed_resets(maximum, expected):
    service = SimulationService(StateOnlyBackend())
    result = service.step_and_reset(fused_payload(reset(service), maximum))
    assert [row["env_id"] for row in result["reset_items"]] == expected
    assert service.slots[0]["done"] is (not expected)


@pytest.mark.parametrize("key,value", [("terminal_observation_mode", "full"), ("reset_seed", True),
                                      ("reset_options", []), ("max_active_envs", -1),
                                      ("max_active_envs", 3), ("max_active_envs", True)])
def test_fused_service_invalid_request_rejected_before_physics(key, value):
    backend = StateOnlyBackend()
    service = SimulationService(backend)
    request = fused_payload(reset(service))
    request[key] = value
    with pytest.raises(ValueError):
        service.step_and_reset(request)
    assert backend.count == [0, 0] and service.fault is None


@pytest.mark.parametrize("failure", ["reset_failure", "bad_reset_ids"])
def test_fused_backend_failure_faults_session_without_partial_protocol_commit(failure):
    backend = StateOnlyBackend()
    service = SimulationService(backend)
    initial = reset(service)
    prior_slots = deepcopy(service.slots)
    original = backend.step_and_reset
    def wrong(items, **kwargs):
        response = original(items, **kwargs)
        if failure == "reset_failure":
            raise RuntimeError("reset renderer failed after physics")
        response["reset_items"][0]["env_id"] = 1
        return response
    backend.step_and_reset = wrong
    request = fused_payload(initial)
    with pytest.raises(RuntimeError):
        service.step_and_reset(request)
    assert service.fault.startswith("step_and_reset:")
    assert service.slots == prior_slots
    before = deepcopy(backend.calls)
    with pytest.raises(RuntimeError, match="unknown"):
        service.step_and_reset(request)
    assert backend.calls == before
