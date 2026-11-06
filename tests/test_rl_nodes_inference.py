"""Generic frozen inference boundary tests without model weights or a GPU."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from pressb.rl_nodes.contract import encode_image
from pressb.rl_nodes.inference import InferenceService


class Backend:
    def __init__(self, horizon=32, action_dim=7, latent_dim=20):
        self.horizon, self.action_dim, self.latent_dim = horizon, action_dim, latent_dim
        self.calls = []
        self.bad_actions = False
        self.closed = False

    def describe(self):
        return dict(
            schema_version=2, node_type="inference", model_id="test-frozen-v1", feature_dim=3,
            capabilities=["encode", "decode", "infer", "external_initial_noise", "features", "context_cache"],
            observation=dict(state=dict(names=["x", "y"], units=["m", "m"]), cameras=["up"]),
            action=dict(representation="joint_position", frame="robot_base",
                        names=[f"j{i}" for i in range(self.action_dim)], units=["rad"] * self.action_dim,
                        horizon=self.horizon, control_hz=30),
            noise=dict(shape=[self.horizon, self.latent_dim]),
        )

    def encode_batch(self, observations, seeds):
        self.calls.append(("encode", len(observations)))
        return [(dict(state=np.asarray(obs["state"], dtype=np.float32),
                      noise=np.random.default_rng(seed).normal(size=(self.horizon, self.latent_dim))),
                 np.array([*obs["state"], seed % 11], dtype=np.float32))
                for obs, seed in zip(observations, seeds)]

    def decode_batch(self, contexts, initial_noises):
        self.calls.append(("decode", len(contexts)))
        result = []
        for context, noise in zip(contexts, initial_noises):
            z = context["noise"] if noise is None else noise
            # Padded latent channels deliberately influence real commands.
            action = z[:, :self.action_dim] + z.sum(axis=1, keepdims=True) + context["state"].sum()
            result.append(action if not self.bad_actions else action[:, :-1])
        return result

    def close(self):
        self.closed = True


def observation(x=0):
    return dict(task="Put the cube into the box.", state=[x, 0.2],
                images={"up": encode_image(np.zeros((4, 6, 3), dtype=np.uint8))},
                model_inputs={})


def request(count=1, include=False):
    return dict(observations=[observation(index) for index in range(count)],
                seeds=list(range(41, 41 + count)), include_base_actions=include)


@pytest.mark.parametrize("shape", [(32, 7, 20), (7, 9, 9)])
def test_shapes_features_baselines_and_cache_isolation(shape):
    backend = Backend(*shape)
    service = InferenceService(backend, batch_size=2)
    encoded = service.encode(request(3, include=True))["items"]
    assert backend.calls == [("encode", 2), ("encode", 1), ("decode", 2), ("decode", 1)]
    assert np.asarray(encoded[0]["base_actions"]).shape == shape[:2]
    np.testing.assert_array_equal(encoded[1]["features"], [1, np.float32(.2), 42 % 11])
    baseline = service.decode(dict(items=[dict(context_id=item["context_id"]) for item in encoded]))["items"]
    for before, after in zip(encoded, baseline):
        np.testing.assert_array_equal(before["base_actions"], after["actions"])
    zero = np.zeros((shape[0], shape[2]), dtype=np.float32)
    controlled = service.decode(dict(items=[dict(context_id=item["context_id"], initial_noise=zero.tolist())
                                           for item in reversed(encoded)]))["items"]
    assert [item["context_id"] for item in controlled] == [item["context_id"] for item in reversed(encoded)]
    np.testing.assert_allclose(controlled[0]["actions"], 2.2)
    np.testing.assert_allclose(controlled[2]["actions"], .2)
    if shape[2] > shape[1]:
        zero[:, -1] = 2
        padded = service.decode(dict(items=[dict(context_id=encoded[0]["context_id"], initial_noise=zero.tolist())]))
        np.testing.assert_allclose(padded["items"][0]["actions"], 2.2)


@pytest.mark.parametrize("fault", ["state", "image", "seed", "include", "capacity"])
def test_all_inputs_are_validated_before_any_model_computation(fault):
    backend = Backend()
    service = InferenceService(backend, batch_size=1, cache_size=3)
    payload = request(3)
    if fault == "state":
        payload["observations"][-1]["state"][0] = float("nan")
    elif fault == "image":
        payload["observations"][-1]["images"]["up"]["data"] = "not png"
    elif fault == "seed":
        payload["seeds"][-1] = True
    elif fault == "include":
        payload["include_base_actions"] = 1
    else:
        payload = request(4)
    with pytest.raises(ValueError):
        service.encode(payload)
    assert not backend.calls
    assert not service.contexts


def test_decode_validation_cannot_partially_compute():
    backend = Backend()
    service = InferenceService(backend, batch_size=1)
    keys = [item["context_id"] for item in service.encode(request(2))["items"]]
    backend.calls.clear()
    with pytest.raises(ValueError):
        service.decode(dict(items=[dict(context_id=keys[0]),
                                   dict(context_id=keys[1], initial_noise=np.zeros((32, 7)).tolist())]))
    with pytest.raises(ValueError):
        service.decode(dict(items=[dict(context_id=keys[0]), dict(context_id="unknown")]))
    assert not backend.calls


def test_failed_model_output_does_not_evict_existing_contexts():
    backend = Backend()
    service = InferenceService(backend, cache_size=2)
    existing = service.encode(request(2))["items"]
    original = list(service.contexts)
    backend.bad_actions = True
    with pytest.raises(ValueError):
        service.encode(request(2, include=True))
    assert list(service.contexts) == original
    assert service.encoded_count == 2
    assert original == [item["context_id"] for item in existing]


def test_ttl_eviction_release_and_infer_do_not_leak_contexts():
    now = [0.0]
    service = InferenceService(Backend(), cache_size=2, cache_ttl=5, clock=lambda: now[0])
    first = service.encode(request())["items"][0]["context_id"]
    now[0] = 1
    second = service.encode(request())["items"][0]["context_id"]
    before = list(service.contexts)
    infer_payload = request(2)
    del infer_payload["include_base_actions"]
    inferred = service.infer(infer_payload)
    assert len(inferred["items"]) == 2
    assert list(service.contexts) == before
    service.decode(dict(items=[dict(context_id=first)]))
    now[0] = 5
    assert service.health()["cache"]["size"] == 1
    with pytest.raises(ValueError, match="expired"):
        service.decode(dict(items=[dict(context_id=first)]))
    assert service.release(dict(context_ids=[second, second])) == {"released": 1}
    assert service.release(dict(context_ids=[second])) == {"released": 0}
    keys = [item["context_id"] for item in service.encode(request(2))["items"]]
    third = service.encode(request())["items"][0]["context_id"]
    assert list(service.contexts) == [keys[1], third]
    service.close()
    service.close()
    assert service.backend.closed and not service.contexts
    with pytest.raises(RuntimeError, match="closed"):
        service.health()


def test_one_shot_external_noise_matches_cached_path_without_eviction():
    service = InferenceService(Backend())
    payload = request(2)
    keys = [item["context_id"] for item in service.encode(payload)["items"]]
    noises = [np.full((32, 20), index, dtype=np.float32).tolist() for index in range(2)]
    decoded = service.decode(dict(items=[dict(context_id=key, initial_noise=noise)
                                         for key, noise in zip(keys, noises)]))
    payload.pop("include_base_actions")
    inferred = service.infer(dict(payload, initial_noises=noises))
    for cached, fresh in zip(decoded["items"], inferred["items"]):
        np.testing.assert_array_equal(cached["actions"], fresh["actions"])
    assert list(service.contexts) == keys


def test_health_returns_independent_metadata_copies():
    service = InferenceService(Backend())
    response = service.health()
    response["noise"]["shape"][1] = 99
    assert service.health()["noise"]["shape"] == [32, 20]


def test_vlajepa_adapter_retains_seeded_sampler_draw_and_raw8_contract():
    torch = pytest.importorskip("torch")
    from pressb.rl_nodes.vlajepa import VLAJEPABackend

    class Frozen:
        action_horizon, action_dim, feature_dim = 7, 9, 3

        def health(self):
            return dict(checkpoint_sha256="a" * 64, checkpoint_step=10600, frozen=True,
                        controller_gripper_width_m=.008)

        def state_preparer(self, value):
            assert value.shape == (8,)

        def encode_batch(self, observations, *, seeds, include_base_actions):
            assert include_base_actions is False
            return [(SimpleNamespace(tokens=torch.zeros((1, 32, 3)), feature=np.asarray(obs["state"][:3])), None)
                    for obs in observations]

        def decode_batch(self, contexts, noises):
            return np.asarray(noises)

    backend = VLAJEPABackend(Frozen())
    image = encode_image(np.zeros((4, 4, 3), dtype=np.uint8))
    obs = dict(task="Press 31 floor.", state=[.1, .2, .3, 1, 0, 0, 0, .008],
               images={"global": image, "wrist": image})
    rng = torch.random.get_rng_state().clone()
    encoded = backend.encode_batch([obs, obs], [18, 72])
    assert torch.equal(rng, torch.random.get_rng_state())
    for (context, feature), seed in zip(encoded, [18, 72]):
        expected = torch.randn((1, 7, 9), generator=torch.Generator().manual_seed(seed))[0].numpy()
        actual = backend.decode_batch([context], [None])[0]
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(feature, [.1, .2, .3])
        backend.decode_batch([context], [np.zeros((7, 9))])
        np.testing.assert_array_equal(backend.decode_batch([context], [None])[0], expected)
    assert backend.describe()["observation"]["state"]["names"][3:7] == ["qw", "qx", "qy", "qz"]
    bad = deepcopy(obs)
    bad["task"] = "Press the red button."
    with pytest.raises(ValueError, match="elevator"):
        backend.validate_observations([bad])
