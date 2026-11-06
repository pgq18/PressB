"""The Wall-OSS/SI boundary without loading large checkpoint weights."""
import base64
import io
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from pressb.rl_nodes.walloss import WallOSSBackend, PIPER_NAMES, PIPER_UNITS, _configure_cuda_budget, _flow_prefix_option, _flow_prefix_mode


def png():
    stream = io.BytesIO()
    Image.new("RGB", (4, 3), color="red").save(stream, format="PNG")
    return {"encoding": "png_base64", "data": base64.b64encode(stream.getvalue()).decode()}


class TinyPolicy:
    def __init__(self):
        self.encode_batch_sizes = []
        self.decode_batch_sizes = []
        self.last_noises = None

    def encode_batch(self, observations, **kwargs):
        self.encode_batch_sizes.append(len(observations))
        result = []
        for obs in observations:
            marker = float(obs["state"][0])
            # Different prompts have different sequence lengths. The adapter
            # passes these contexts together to the native batch implementation.
            sequence_length = len(obs["prompt"].split()) + 2
            result.append({"inputs": {"inputs_embeds": torch.full((1, sequence_length, 8), marker)},
                           "feature": torch.full((1, 8), marker)})
        return result

    def decode_batch(self, contexts, initial_noise=None):
        self.decode_batch_sizes.append(len(contexts))
        self.last_noises = initial_noise.clone()
        markers = torch.tensor([context["feature"][0, 0] for context in contexts])[:, None, None]
        return {"action": (initial_noise[:, :, :7].float() + markers).numpy()}

    def encode(self, *args, **kwargs):
        raise AssertionError("The adapter must use native batched encoding")

    def decode(self, *args, **kwargs):
        raise AssertionError("The adapter must use native batched decoding")


@pytest.fixture
def backend():
    result = WallOSSBackend.__new__(WallOSSBackend)
    result.dataset_name = "lerobot/piper"
    result.dimension = 7
    result.horizon = 32
    result.noise_dim = 20
    result.feature_dim = 8
    result.state_scale = np.array([180 / np.pi] * 6 + [100])
    result.action_scale = np.array([np.pi / 180] * 6 + [.01])
    result.state_min = np.zeros(7)
    result.state_delta = np.array([180.] * 6 + [10.])
    result.camera_mapping = {"up": "face_view", "wrist": "left_wrist_view"}
    result.policy = TinyPolicy()
    result.num_inference_timesteps = 10
    result.model_id = "tiny_wall"
    result.names, result.units = PIPER_NAMES, PIPER_UNITS
    result.control_hz, result.stats_path = 30., "tiny.json"
    for name in ("weights_sha256", "normalization_sha256", "model_config_sha256", "source_revision"):
        setattr(result, name, "synthetic")
    return result


def observation():
    return {"state": [np.pi / 2] * 6 + [.05], "images": {"up": png(), "wrist": png()},
            "task": "Put the red cube into the box.", "model_inputs": {}}


def test_si_input_normalization_and_output_units(backend):
    prepared = backend._prepare_observation(observation())
    np.testing.assert_allclose(prepared["state"], 0, atol=1e-7)
    assert prepared["face_view"].shape == (3, 4, 3)
    assert prepared["dataset_names"] == ["lerobot/piper"]
    (context, features), = backend.encode_batch([observation()], [42])
    output, = backend.decode_batch([context], [np.ones((32, 20))])
    np.testing.assert_allclose(output[:, :6], np.pi / 180)
    np.testing.assert_allclose(output[:, 6], .01)
    assert features.shape == (8,)


def test_baseline_seed_independent_of_batch_grouping_and_other_decodes(backend):
    batch = backend.encode_batch([observation(), observation()], [12, 99])
    single = backend.encode_batch([observation()], [99])
    contexts = [item[0] for item in batch]
    expected = backend.decode_batch(contexts, [None, None])
    backend.decode_batch(contexts, [np.zeros((32, 20)), np.ones((32, 20))])
    repeated = backend.decode_batch(contexts, [None, None])
    only, = backend.decode_batch([single[0][0]], [None])
    np.testing.assert_array_equal(expected[0], repeated[0])
    np.testing.assert_array_equal(expected[1], repeated[1])
    np.testing.assert_array_equal(expected[1], only)


def test_invalid_dimensions_and_unknown_model_inputs_fail(backend):
    (context, _), = backend.encode_batch([observation()], [1])
    for value in (np.zeros((32, 7)), np.full((32, 20), np.nan), np.zeros((32, 20), dtype=int)):
        with pytest.raises(ValueError):
            backend.decode_batch([context], [value])
    bad = observation()
    bad["model_inputs"] = {"dataset_name": "wrong"}
    with pytest.raises(ValueError):
        backend.encode_batch([bad], [1])
    bad["model_inputs"] = {"silently_ignored_option": 123}
    with pytest.raises(ValueError):
        backend.encode_batch([bad], [1])


def test_wall_backend_through_generic_service(backend):
    from pressb.rl_nodes.inference import InferenceService
    service = InferenceService(backend, batch_size=2)
    health = service.health()
    assert health["noise"]["shape"] == [32, 20]
    assert health["action"]["units"] == ["rad"] * 6 + ["m"]
    assert health["batching"] == {"native_model_batching": True, "microbatch_size": 2}
    assert health["microbatch_size"] == 2
    encoded = service.encode({"observations": [observation()], "seeds": [41], "include_base_actions": True})
    item = encoded["items"][0]
    reply = service.decode({"items": [{"context_id": item["context_id"], "initial_noise": np.ones((32, 20)).tolist()}]})
    assert np.shape(reply["items"][0]["actions"]) == (32, 7)
    assert reply["items"][0]["actions"] != item["base_actions"]
    repeated = service.decode({"items": [{"context_id": item["context_id"]}]})
    assert repeated["items"][0]["actions"] == item["base_actions"]
    assert service.release({"context_ids": [item["context_id"]]}) == {"released": 1}


def test_native_batch_calls_follow_service_microbatch_size(backend):
    from pressb.rl_nodes.inference import InferenceService
    service = InferenceService(backend, batch_size=2)
    observations = [observation() for _ in range(5)]
    for index, obs in enumerate(observations):
        obs["task"] = "Put " + "the " * index + "cube in the box."
        obs["state"][0] = index * np.pi / 8
    encoded = service.encode({"observations": observations, "seeds": list(range(5)),
                              "include_base_actions": True})
    assert backend.policy.encode_batch_sizes == [2, 2, 1]
    assert backend.policy.decode_batch_sizes == [2, 2, 1]
    np.testing.assert_allclose([item["features"][0] for item in encoded["items"]],
                               [-1, -.75, -.5, -.25, 0], atol=1e-7)
    # Decode arbitrary cached rows in a different order, including a duplicate,
    # without changing which observation/noise each cache entry owns.
    order = [4, 0, 2, 0, 3]
    decoded = service.decode({"items": [{"context_id": encoded["items"][index]["context_id"]}
                                        for index in order]})
    for item, index in zip(decoded["items"], order):
        np.testing.assert_array_equal(item["actions"], encoded["items"][index]["base_actions"])
    assert backend.policy.encode_batch_sizes == [2, 2, 1]
    assert backend.policy.decode_batch_sizes == [2, 2, 1, 2, 2, 1]


def test_batched_noise_rows_preserve_individual_seeds_and_external_values(backend):
    batch = backend.encode_batch([observation()] * 3, [4, 7, 4])
    contexts = [entry[0] for entry in batch]
    expected_baselines = [torch.randn((1, 32, 20), generator=torch.Generator().manual_seed(seed))
                          for seed in [4, 7, 4]]
    for context, expected in zip(contexts, expected_baselines):
        torch.testing.assert_close(context.baseline_noise, expected, rtol=0, atol=0)
    explicit_noise = np.full((32, 20), .25, dtype=np.float64)
    backend.decode_batch(contexts, [None, explicit_noise, None])
    assert backend.policy.last_noises.shape == (3, 32, 20)
    torch.testing.assert_close(backend.policy.last_noises[0], expected_baselines[0][0], rtol=0, atol=0)
    torch.testing.assert_close(backend.policy.last_noises[1], torch.full((32, 20), .25), rtol=0, atol=0)
    torch.testing.assert_close(backend.policy.last_noises[2], expected_baselines[2][0], rtol=0, atol=0)
    for context, expected in zip(contexts, expected_baselines):
        torch.testing.assert_close(context.baseline_noise, expected, rtol=0, atol=0)
    assert backend.policy.encode_batch_sizes == [3]
    assert backend.policy.decode_batch_sizes == [3]


def test_invalid_later_row_fails_before_any_native_batch_call(backend):
    bad = observation()
    bad["model_inputs"] = {"dataset_name": "wrong"}
    with pytest.raises(ValueError):
        backend.encode_batch([observation(), bad], [1, 2])
    assert backend.policy.encode_batch_sizes == []
    contexts = [item[0] for item in backend.encode_batch([observation()] * 2, [1, 2])]
    with pytest.raises(ValueError):
        backend.decode_batch(contexts, [None, np.zeros((31, 20))])
    assert backend.policy.decode_batch_sizes == []


def test_invalid_native_batch_shape_does_not_commit_service_cache(backend):
    from pressb.rl_nodes.inference import InferenceService
    service = InferenceService(backend, batch_size=2)
    backend.policy.decode_batch = lambda *args, **kwargs: {"action": np.zeros((1, 32, 7))}
    with pytest.raises(ValueError, match="invalid batched actions"):
        service.encode({"observations": [observation()] * 2, "seeds": [1, 2], "include_base_actions": True})
    assert service.health()["cache"]["size"] == 0
    assert service.health()["encoded_count"] == 0


def test_empty_adapter_batch_does_not_call_native_model(backend):
    assert backend.encode_batch([], []) == []
    assert backend.decode_batch([], []) == []
    assert backend.policy.encode_batch_sizes == backend.policy.decode_batch_sizes == []


def test_optional_cuda_budget_rejects_bad_config_before_device_use():
    calls = []
    torch_stub = SimpleNamespace(device=torch.device, cuda=SimpleNamespace(
        set_per_process_memory_fraction=lambda fraction, device: calls.append((fraction, str(device)))))
    assert _configure_cuda_budget({}, torch_stub) is None
    assert calls == []
    for fraction in (0, -1, 1.1, True, float("nan"), "0.38"):
        with pytest.raises(ValueError):
            _configure_cuda_budget({"cuda_memory_fraction": fraction}, torch_stub)
    with pytest.raises(ValueError, match="CUDA"):
        _configure_cuda_budget({"cuda_memory_fraction": .38, "device": "cpu"}, torch_stub)
    assert calls == []
    assert _configure_cuda_budget({"cuda_memory_fraction": .38, "device": "cuda:0"}, torch_stub) == .38
    assert calls == [(.38, "cuda:0")]


def test_prefix_runtime_is_opt_in_and_separate_from_observation_cache(backend):
    assert _flow_prefix_option({}) is False
    assert _flow_prefix_option({"flow_prefix_cache": True}) is True
    for value in (None, 0, 1, "false", "true"):
        with pytest.raises(ValueError, match="boolean"):
            _flow_prefix_option({"flow_prefix_cache": value})
    assert backend.describe()["diffusion_prefix_kv_reuse"] is False
    backend.flow_prefix_cache = True
    metadata = backend.describe()
    assert metadata["diffusion_prefix_kv_reuse"] is True
    assert metadata["diffusion_prefix_cache_scope"] == "within_one_decode"
    assert metadata["transformer_kv_cache"] is False
    assert _flow_prefix_mode({}) == "suffix"
    assert _flow_prefix_mode({"flow_prefix_cache_mode": "full_attention"}) == "full_attention"
    for value in (True, None, "unknown"):
        with pytest.raises(ValueError, match="flow_prefix_cache_mode"):
            _flow_prefix_mode({"flow_prefix_cache_mode": value})
    backend.flow_prefix_cache_mode = "full_attention"
    assert backend.describe()["diffusion_prefix_cache_mode"] == "full_attention"
    assert backend.describe()["diffusion_prefix_reuse"] is True
    assert backend.describe()["diffusion_prefix_kv_reuse"] is False


def test_service_decodes_each_png_once_and_reuses_all_backend_preparation(backend, monkeypatch):
    from pressb.rl_nodes import inference
    original_decode = inference.decode_image
    decoded = []
    prepared = []
    def counted_decode(value, **kwargs):
        decoded.append(value)
        return original_decode(value, **kwargs)
    original_prepare = backend._prepare_observation
    def counted_prepare(value):
        result = original_prepare(value)
        prepared.append(result)
        return result
    monkeypatch.setattr(inference, "decode_image", counted_decode)
    monkeypatch.setattr(backend, "_prepare_observation", counted_prepare)
    def no_second_decode(value):
        pytest.fail("Validated image must be reused during backend preflight and encode")
    monkeypatch.setattr("pressb.rl_nodes.walloss._image", no_second_decode)
    service = inference.InferenceService(backend, batch_size=1, collect_timing=True)
    result = service.encode({"observations": [observation(), observation()], "seeds": [1, 2]})
    assert len(decoded) == 4  # two cameras for each observation, exactly once
    assert prepared[0] is prepared[2] and prepared[1] is prepared[3]
    assert len(result["items"]) == 2
    assert set(result["timing"]) == {"validation_seconds", "backend_preprocess_seconds",
                                      "backend_encode_seconds", "total_seconds"}
    assert all(np.isfinite(v) and v >= 0 for v in result["timing"].values())


def test_bad_last_binary_png_does_not_reach_any_gpu_batch_or_commit(backend):
    from pressb.rl_nodes.inference import InferenceService
    service = InferenceService(backend, batch_size=1)
    bad = observation()
    bad["images"]["wrist"] = {"encoding": "png_bytes", "data": b"not a PNG"}
    with pytest.raises(ValueError):
        service.encode({"observations": [observation(), bad], "seeds": [1, 2]})
    assert backend.policy.encode_batch_sizes == []
    assert service.health()["cache"]["size"] == 0


def test_binary_preprocessed_pixels_preserve_noise_and_features(backend):
    from pressb.rl_nodes.inference import InferenceService
    from pressb.rl_nodes.image_transport import prepare_wire_payload, walloss_resize_spec
    backend.image_preprocessing = walloss_resize_spec(256)
    service = InferenceService(backend, batch_size=2)
    payload = {"observations": [observation(), observation()], "seeds": [1, 2], "include_base_actions": True}
    baseline = service.encode(payload)
    transformed = prepare_wire_payload("/encode", payload, service.health(),
                                       image_transport="auto", preprocess_images=True)
    optimized = service.encode(transformed)
    for original, compact in zip(baseline["items"], optimized["items"]):
        assert original["features"] == compact["features"]
        assert original["base_actions"] == compact["base_actions"]
