"""CPU contracts for the frozen adapter, without checkpoint/GPU dependencies."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
torch = pytest.importorskip("torch")
from torch import nn

from pressb.online_rl.inference import FrozenPiperBackend, InferenceApplication, flow_from_context
from pressb.online_rl.protocol import encode_image, pose8_to_pose9


class TinyActionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(9, 9)

    def forward(self, actions, timesteps):
        return self.linear(actions) + timesteps[:, None, None].float() / 1000


class TinyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(9, 9)
        self.condition = nn.Linear(4, 9)
        self.timesteps = []

    def forward(self, hidden_states, encoder_hidden_states, timestep):
        self.timesteps.append(timestep.tolist())
        return self.hidden(hidden_states) + self.condition(encoder_hidden_states.mean(1))[:, None]


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_horizon, self.action_dim = 7, 9
        self.num_inference_timesteps, self.num_timestep_buckets = 4, 1000
        self.config = SimpleNamespace(add_pos_embed=True)
        self.state_encoder = nn.Linear(9, 9)
        self.action_encoder = TinyActionEncoder()
        self.action_decoder = nn.Linear(9, 9)
        self.future_tokens = nn.Embedding(32, 9)
        self.position_embedding = nn.Embedding(7, 9)
        self.model = TinyDiT()

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _encode_state(self, state, batch_size):
        if state is None:
            return None
        assert state.shape == (batch_size, 1, 9)
        return self.state_encoder(state.to(dtype=self.dtype))


@torch.no_grad()
def legacy_sampler(head, tokens, state):
    """Original randn/Euler operation order, independent of the addon function."""
    tokens = tokens.to(dtype=head.dtype)
    batch_size = tokens.shape[0]
    device = tokens.device
    actions = torch.randn(size=(batch_size, head.action_horizon, head.action_dim),
                          dtype=tokens.dtype, device=device)
    steps = head.num_inference_timesteps
    dt = 1. / steps
    state_features = head._encode_state(state, batch_size)
    for t in range(steps):
        timesteps = torch.full(size=(batch_size,), fill_value=int(t / float(steps) * head.num_timestep_buckets), device=device)
        features = head.action_encoder(actions, timesteps)
        if head.config.add_pos_embed:
            positions = torch.arange(features.shape[1], dtype=torch.long, device=device)
            features = features + head.position_embedding(positions).unsqueeze(0)
        future = head.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
        hidden = torch.cat((state_features, future, features), dim=1) \
            if state_features is not None else torch.cat((future, features), dim=1)
        prediction = head.action_decoder(head.model(hidden_states=hidden, encoder_hidden_states=tokens, timestep=timesteps))
        actions = actions + dt * prediction[:, -head.action_horizon:]
    return actions


class TinyQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.model = SimpleNamespace(config=SimpleNamespace(hidden_size=4))
        self.calls = 0
        self.prepared = []

    def build_qwenvl_inputs(self, images, instructions, prompt_replace_dict):
        self.prepared.append((images, instructions, prompt_replace_dict))
        return {"input_ids": torch.tensor([[1] + [9] * 32] * len(instructions))}

    def forward(self, input_ids, **kwargs):
        self.calls += 1
        assert not self.training and not torch.is_grad_enabled()
        return SimpleNamespace(hidden_states=(self.embedding(input_ids),))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.qwen_vl_interface = TinyQwen()
        self.action_model = TinyHead()
        self.config = SimpleNamespace(
            framework=SimpleNamespace(vj2_model=SimpleNamespace(num_embodied_action_tokens_per_instruction=32)),
            datasets=SimpleNamespace(vla_data=SimpleNamespace()),
        )
        self.embodied_action_token_id = 9
        self.replace_prompt, self.embodied_replace_prompt = "<dynamic>", "<embodied>"


def prepare_state(state):
    return pose8_to_pose9(state).astype(np.float32).reshape(1, 1, 9)


@pytest.fixture
def backend():
    torch.manual_seed(84)
    model = TinyModel()
    policy = SimpleNamespace(model=model, device=torch.device("cpu"), resolution=224,
                             controller_gripper_width_m=.008,
                             provenance=dict(checkpoint="/run/checkpoints/step_010600", model_sha256="a" * 64))
    return FrozenPiperBackend(policy, state_preparer=prepare_state)


def observation():
    return dict(task="Press 24 floor.", state=[.1, .2, .3, 1., 0., 0., 0., .008],
                control_state=[0.],
                images={"global": encode_image(np.full((480, 640, 3), [230, 0, 0], dtype=np.uint8)),
                        "wrist": encode_image(np.full((480, 640, 3), [0, 220, 0], dtype=np.uint8))})


def encode_request(*seeds, include=False):
    return dict(observations=[observation() for _ in seeds], seeds=list(seeds), include_base_actions=include,
                protocol_version=1)


def decode_request(context_id, noise=None):
    return dict(items=[dict(context_id=context_id, initial_noise=(np.zeros((7, 9)) if noise is None else noise).tolist())],
                protocol_version=1)


@pytest.mark.parametrize("with_state", [True, False])
def test_explicit_noise_matches_original_four_step_euler(backend, with_state):
    head = backend.model.action_model
    tokens = torch.randn(2, 32, 4).bfloat16()
    state = torch.randn(2, 1, 9) if with_state else None
    torch.manual_seed(923)
    expected = legacy_sampler(head, tokens, state)
    torch.manual_seed(923)
    noise = torch.randn(2, 7, 9)
    head.model.timesteps.clear()
    actual = flow_from_context(head, tokens, state, noise)
    assert torch.equal(actual, expected)
    assert head.model.timesteps == [[0, 0], [250, 250], [500, 500], [750, 750]]
    assert not actual.requires_grad
    # Explicit noise consumes no random numbers and is not modified in place.
    before = torch.random.get_rng_state()
    retained = noise.clone()
    assert torch.equal(flow_from_context(head, tokens, state, noise), actual)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert torch.equal(retained, noise)


def test_baseline_seed_matches_legacy_and_model_is_frozen(backend):
    application = InferenceApplication(backend)
    rng = torch.random.get_rng_state()
    item = application.encode(encode_request(912, include=True))["items"][0]
    assert torch.equal(torch.random.get_rng_state(), rng)
    context = application.contexts[item["context_id"]][1]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(912)
        expected = legacy_sampler(backend.model.action_model, context.tokens, context.state)
    np.testing.assert_array_equal(item["base_actions_pose9"], expected[0].numpy())
    assert all(not parameter.requires_grad for parameter in backend.model.parameters())
    assert all(not module.training for module in backend.model.modules())
    assert context.tokens.shape == (1, 32, 4)
    assert context.state.shape == (1, 1, 9)
    np.testing.assert_array_equal(item["feature"], context.tokens.float().mean(1)[0].numpy())
    qwen = backend.model.qwen_vl_interface
    images, tasks, replacements = qwen.prepared[0]
    assert tasks == ["Press 24 floor."]
    assert images[0][0].getpixel((0, 0)) == (230, 0, 0)
    assert images[0][1].getpixel((0, 0)) == (0, 220, 0)
    assert all(image.size == (224, 224) for image in images[0])
    assert replacements == {"{actions}": "<dynamic>", "{e_actions}": "<embodied>"}
    health = application.health()
    assert health["frozen"] and health["checkpoint_step"] == 10600
    assert health["checkpoint_sha256"] == "a" * 64


def test_per_sample_seed_is_independent_of_request_grouping(backend):
    app = InferenceApplication(backend)
    batch = app.encode(encode_request(123, 456, include=True))["items"]
    singles = [app.encode(encode_request(seed, include=True))["items"][0] for seed in (123, 456)]
    for together, single in zip(batch, singles):
        assert together["base_actions_pose9"] == single["base_actions_pose9"]
        assert together["feature"] == single["feature"]


def test_decode_uses_cached_full_context_and_never_reencodes(backend):
    app = InferenceApplication(backend)
    key = app.encode(encode_request(5))["items"][0]["context_id"]
    assert backend.model.qwen_vl_interface.calls == 1
    first = app.decode(decode_request(key))
    assert first == app.decode(decode_request(key))
    assert backend.model.qwen_vl_interface.calls == 1
    changed = app.decode(decode_request(key, np.ones((7, 9))))
    assert changed["items"][0]["actions_pose9"] != first["items"][0]["actions_pose9"]
    np.testing.assert_allclose(np.array(first["items"][0]["actions_pose8"])[:, 7], .008)
    assert app.release({"context_ids": [key], "protocol_version": 1}) == {"released": 1}
    with pytest.raises(ValueError, match="missing or expired"):
        app.decode(decode_request(key))
    assert backend.model.qwen_vl_interface.calls == 1


def test_cache_capacity_ttl_and_new_batch_are_bounded(backend):
    now = [0.]
    app = InferenceApplication(backend, cache_size=2, cache_ttl=10., clock=lambda: now[0])
    old = app.encode(encode_request(1))["items"][0]["context_id"]
    new = app.encode(encode_request(2, 3))["items"]
    assert len(app.contexts) == 2 and old not in app.contexts
    app.decode(decode_request(new[0]["context_id"]))
    now[0] = 10.
    with pytest.raises(ValueError, match="missing or expired"):
        app.decode(decode_request(new[0]["context_id"]))
    assert app.health()["cache"]["size"] == 0
    assert app.release({"context_ids": [old]}) == {"released": 0}


def test_validate_entire_batch_before_encoder_and_decode_calls(backend):
    app = InferenceApplication(backend, cache_size=2)
    request = encode_request(1, 2)
    request["observations"][1]["state"] = [0] * 8
    with pytest.raises(ValueError):
        app.encode(request)
    assert backend.model.qwen_vl_interface.calls == 0
    assert len(app.contexts) == 0
    key = app.encode(encode_request(4))["items"][0]["context_id"]
    request = decode_request(key)
    request["items"].append(dict(context_id="missing", initial_noise=np.zeros((7, 9)).tolist()))
    backend.model.action_model.model.timesteps.clear()
    with pytest.raises(ValueError, match="missing or expired"):
        app.decode(request)
    assert backend.model.action_model.model.timesteps == []


@pytest.mark.parametrize("seed", [True, -1, 2**63, 1.5])
def test_invalid_seeds_never_call_encoder(backend, seed):
    app = InferenceApplication(backend)
    with pytest.raises(ValueError, match="seeds"):
        app.encode(encode_request(seed))
    assert backend.model.qwen_vl_interface.calls == 0


@pytest.mark.parametrize("noise", [[[0.] * 9] * 6, [[True] * 9] * 7, [[float("nan")] * 9] * 7])
def test_bad_noise_is_rejected_before_sampling(backend, noise):
    app = InferenceApplication(backend)
    key = app.encode(encode_request(1))["items"][0]["context_id"]
    with pytest.raises(ValueError, match="initial_noise"):
        app.decode(dict(items=[dict(context_id=key, initial_noise=noise)]))
    assert backend.model.action_model.model.timesteps == []


def test_response_pose_conversion_reuses_original_policy_hook(backend):
    seen = []
    converter = backend.pose_converter
    backend.pose_converter = lambda actions: (seen.append(actions.copy()) or converter(actions))
    app = InferenceApplication(backend)
    key = app.encode(encode_request(1, include=True))["items"][0]["context_id"]
    app.decode(decode_request(key))
    assert len(seen) == 2


def test_batch_encoder_and_flow_preserve_per_sample_noise_without_serial_forwards(backend):
    serial = InferenceApplication(backend)
    seeds = [13, 47, 98, 13, 7]
    expected = serial.encode(encode_request(*seeds, include=True))["items"]
    qwen = backend.model.qwen_vl_interface
    qwen.calls = 0
    head = backend.model.action_model.model
    head.timesteps.clear()
    before_rng = torch.random.get_rng_state()
    batched = InferenceApplication(backend, batch_size=3)
    actual = batched.encode(encode_request(*seeds, include=True))["items"]
    assert torch.equal(torch.random.get_rng_state(), before_rng)
    assert qwen.calls == 2
    assert head.timesteps == [[0] * 3, [250] * 3, [500] * 3, [750] * 3,
                              [0] * 2, [250] * 2, [500] * 2, [750] * 2]
    for reference, result in zip(expected, actual):
        np.testing.assert_array_equal(result["feature"], reference["feature"])
        np.testing.assert_allclose(result["base_actions_pose9"], reference["base_actions_pose9"], atol=4e-7)
    assert batched.health()["encoder_batch_mode"] == "batched"
    assert batched.health()["microbatch_size"] == 3
    assert all(not parameter.requires_grad for parameter in backend.model.parameters())


def test_batch_decode_uses_full_cached_tokens_and_validates_all_before_forward(backend):
    app = InferenceApplication(backend, batch_size=2)
    items = app.encode(encode_request(2, 4, 7))["items"]
    requests = [decode_request(item["context_id"], np.full((7, 9), index))["items"][0]
                for index, item in enumerate(items)]
    expected = [backend.decode(app.contexts[item["context_id"]][1], item["initial_noise"]) for item in requests]
    head = backend.model.action_model.model
    head.timesteps.clear()
    actual = app.decode(dict(items=requests))["items"]
    assert backend.model.qwen_vl_interface.calls == 2
    assert head.timesteps == [[0, 0], [250, 250], [500, 500], [750, 750], [0], [250], [500], [750]]
    for reference, result in zip(expected, actual):
        np.testing.assert_allclose(result["actions_pose9"], reference, atol=5e-7)
    head.timesteps.clear()
    with pytest.raises(ValueError, match="missing or expired"):
        app.decode(dict(items=requests + [decode_request("missing")["items"][0]]))
    assert head.timesteps == []


def test_encode_release_is_committed_only_after_success(backend):
    app = InferenceApplication(backend, batch_size=4)
    old = app.encode(encode_request(1))["items"][0]["context_id"]
    invalid = encode_request(2)
    invalid.update(release_context_ids=[old])
    invalid["observations"][0]["state"] = [0] * 8
    with pytest.raises(ValueError):
        app.encode(invalid)
    assert old in app.contexts
    new = encode_request(3, 4)
    new.update(release_context_ids=[old, old, "already-expired"])
    response = app.encode(new)
    assert old not in app.contexts
    assert len(app.contexts) == len(response["items"]) == 2


def test_pre_resized_transport_preserves_encoder_pixels(backend):
    import base64
    from io import BytesIO
    from PIL import Image

    app = InferenceApplication(backend, batch_size=2)
    request = encode_request(12, 12, include=True)
    for camera, encoded in request["observations"][1]["images"].items():
        image = Image.open(BytesIO(base64.b64decode(encoded)))
        buffer = BytesIO()
        image.resize((224, 224), Image.Resampling.BILINEAR).save(buffer, format="PNG")
        request["observations"][1]["images"][camera] = base64.b64encode(buffer.getvalue()).decode()
    result = app.encode(request)["items"]
    prepared = backend.model.qwen_vl_interface.prepared[-1][0]
    for first, second in zip(prepared[0], prepared[1]):
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    np.testing.assert_allclose(result[0]["base_actions_pose9"], result[1]["base_actions_pose9"], atol=5e-7)


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_invalid_microbatch_is_rejected(backend, size):
    with pytest.raises(ValueError, match="batch_size"):
        InferenceApplication(backend, batch_size=size)


def test_packed_features_are_exact_little_endian_float32_and_opt_in(backend):
    import base64

    app = InferenceApplication(backend, batch_size=2)
    reference = app.encode(encode_request(3, 8))["items"]
    packed = app.encode(dict(**encode_request(3, 8), feature_encoding="float32_base64"))["items"]
    assert app.health()["feature_encodings"] == ["json", "float32_base64"]
    for first, second in zip(reference, packed):
        assert "feature" in first and "feature_f32_b64" not in first
        assert "feature" not in second and second["feature_dim"] == backend.feature_dim
        raw = base64.b64decode(second["feature_f32_b64"], validate=True)
        assert len(raw) == 4 * backend.feature_dim
        np.testing.assert_array_equal(np.frombuffer(raw, dtype="<f4"), first["feature"])
        assert first["state_pose9"] == second["state_pose9"]


def test_bad_feature_encoding_is_rejected_before_any_encoder_or_release(backend):
    app = InferenceApplication(backend, batch_size=2)
    old = app.encode(encode_request(2))["items"][0]["context_id"]
    calls = backend.model.qwen_vl_interface.calls
    with pytest.raises(ValueError, match="feature_encoding"):
        app.encode(dict(**encode_request(3), feature_encoding="float16", release_context_ids=[old]))
    assert old in app.contexts
    assert backend.model.qwen_vl_interface.calls == calls


@pytest.mark.parametrize("relocated", [False, True])
def test_loader_forwards_backbone_paths_without_rewriting_run_config(backend, monkeypatch, tmp_path, relocated):
    import pressb.online_rl.inference as inference

    repo = tmp_path / "VLA-JEPA"
    source = repo / "starVLA/inference/piper_policy.py"
    source.parent.mkdir(parents=True)
    source.write_text("# Fake verified-loader module for argument forwarding only.\n")
    checkpoint = tmp_path / "run/checkpoints/step_010600"
    kwargs = dict(base_vlm=tmp_path / "Qwen3-VL-2B-Instruct", base_encoder=tmp_path / "vjepa2-config") \
        if relocated else {}
    calls = []

    def policy_loader(path, **options):
        calls.append((path, options))
        return backend.policy

    module = SimpleNamespace(__file__=str(source), PiperPolicy=policy_loader,
                             prepare_state=prepare_state, pose9_to_pose8=backend.pose_converter)
    backend.model.qwen_vl_interface.model.config.hidden_size = 2048
    monkeypatch.setattr(inference.sys, "path", inference.sys.path.copy())
    monkeypatch.setattr(inference.importlib, "import_module", lambda name: module)
    monkeypatch.setattr(inference, "_source_identity", lambda path: dict(vla_repo=str(path)))
    loaded = FrozenPiperBackend.load(repo, checkpoint, device="cpu", **kwargs)
    assert calls == [(checkpoint, dict(device="cpu", base_vlm=kwargs.get("base_vlm"),
                                      base_encoder=kwargs.get("base_encoder")))]
    assert loaded.feature_dim == 2048 and loaded.token_count == 32
    assert all(not parameter.requires_grad for parameter in loaded.model.parameters())
