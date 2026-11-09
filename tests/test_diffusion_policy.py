"""Actual visual encoder/UNet/optimizer and serving contract regression tests."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from pressb.model_sources.diffusion_policy.policy import VisualDiffusionPolicy
from pressb.training.diffusion_policy import load_checkpoint
from pressb.models.diffusion_policy import DiffusionPolicyBackend


def configuration():
    return dict(horizon=8, diffusion_steps=10, inference_steps=5, unet_dim=8,
                unet_dim_mults=[1, 2], time_embedding_dim=8, keypoints=2,
                task_embedding_dim=2, state_names=["joint1", "joint2"],
                state_units=["rad", "rad"], action_names=["joint1", "joint2"],
                action_units=["rad", "rad"], cameras=["up", "wrist"],
                tasks=["Press the red button."], resolution=64, control_hz=30)


def policy():
    torch.manual_seed(2)
    return VisualDiffusionPolicy(configuration(), dict(state_center=[.1, .2], state_scale=[1, 2],
                                                       action_center=[.3, .4], action_scale=[2, 3]))


def checkpoint(path, model):
    normalizer = {key: getattr(model, key).tolist() for key in ("state_center", "state_scale", "action_center", "action_scale")}
    torch.save(dict(format="pressb.diffusion_policy.v1", model_config=model.config, normalization=normalizer,
                    model=model.state_dict(), ema=model.state_dict(), step=1,
                    training_contract={"dataset_sha256": "a" * 64}), path)


def test_real_visual_supervision_optimizer_and_roundtrip(tmp_path):
    model = policy()
    images = torch.randint(0, 255, (2, 2, 3, 64, 64), dtype=torch.uint8)
    states = torch.zeros(2, 2)
    task = torch.zeros(2, dtype=torch.long)
    actions = torch.zeros(2, 8, 2)
    before = model.diffusion.network.final_conv[-1].weight.detach().clone()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = model.loss(images, states, task, actions)
    loss.backward()
    assert model.encoder.backbone[0].weight.grad.abs().sum() > 0
    opt.step()
    assert not torch.equal(before, model.diffusion.network.final_conv[-1].weight)
    path = tmp_path / "policy.pt"
    checkpoint(path, model)
    restored, _ = load_checkpoint(path)
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())


def test_ddpm_transition_matches_upstream_equations_and_head_only():
    model = policy()
    expert = model.action_expert({"kernel": "ddpm", "denoising_steps": 10, "min_std": .001})
    x, features = torch.randn(2, 8, 2), torch.randn(2, model.feature_dim)
    for step in (0, 4, 9):
        mean, std = expert.transition(x, step, {"features": features})
        native_mean, native_logvar = model.diffusion.p_mean_var(x, torch.full((2,), 9-step), {"state": features})
        torch.testing.assert_close(mean, native_mean, atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(std, native_logvar.mul(.5).exp().clamp_min(.001).expand_as(x))
    assert not any(name.startswith("encoder") for name, _ in expert.named_parameters())
    assert set(map(id, expert.parameters())) == set(map(id, model.diffusion.network.parameters()))


def test_ddim_has_finite_positive_terminal_sigma_and_changes_conditioning():
    model = policy()
    expert = model.action_expert({"kernel": "ddim", "denoising_steps": 5, "eta": .5, "min_std": .02})
    x, features = torch.randn(2, 8, 2), torch.randn(2, model.feature_dim)
    mean, std = expert.transition(x, torch.tensor([0, 4]), {"features": features})
    assert torch.isfinite(mean).all() and (std >= .02).all()
    changed, _ = expert.transition(x, torch.tensor([0, 4]), {"features": features + 2})
    assert not torch.equal(mean, changed)
    mean.sum().backward()
    assert any(p.grad is not None for p in expert.parameters())


def test_backend_batch_noise_and_context_reuse(tmp_path):
    from pressb.runtime.contract import encode_image
    path = tmp_path / "policy.pt"
    checkpoint(path, policy())
    backend = DiffusionPolicyBackend({"checkpoint": str(path), "device": "cpu"})
    image = encode_image(np.full((64, 64, 3), 80, dtype=np.uint8))
    row = {"state": [.1, .2], "task": "Press the red button.", "images": {"up": image, "wrist": image}}
    contexts = backend.encode_batch([row, row], [5, 7])
    outputs = backend.decode_batch([c[0] for c in contexts], [None, None])
    first = backend.encode_batch([row], [5])
    single = backend.decode_batch([first[0][0]], [None])[0]
    np.testing.assert_allclose(outputs[0], single, atol=1e-5)
    other = backend.decode_batch([first[0][0]], [np.zeros((8, 2), np.float32)])[0]
    assert not np.allclose(other, single)
    assert backend.describe()["action"]["units"] == ["rad", "rad"]
    assert backend.pack_contexts([first[0][0]])["features"].shape == (1, backend.feature_dim)
    raw_backend = DiffusionPolicyBackend({"checkpoint": str(path), "device": "cpu", "use_ema": False})
    assert raw_backend.describe()["checkpoint_variant"] == "model"
    assert raw_backend.describe()["model_config_sha256"] != backend.describe()["model_config_sha256"]
    with pytest.raises(ValueError, match="vocabulary"):
        backend.encode_batch([{**row, "task": "Unseen task"}], [5])


def test_bundled_upstream_files_match_preserved_manifest():
    from pressb.model_sources import builtin_source_root
    root = builtin_source_root("diffusion_policy")
    manifest = json.loads((root / "source_manifest.json").read_text())
    assert manifest["commit"] == "cc7234ad7ff39a8f32de3af903606723a16f0648"
    for entry in manifest["files"]:
        assert hashlib.sha256((root / entry["path"]).read_bytes()).hexdigest() == entry["bundled_sha256"]
    assert "MIT License" in (root / "LICENSE").read_text()


def test_actual_diffusion_backend_chain_service_and_trainable_head(tmp_path):
    from pressb.runtime.contract import encode_image
    from pressb.runtime.inference import InferenceService
    from pressb.runtime.expert_sync import tree_from_wire
    from pressb.runtime.denoising import GaussianChainPolicy

    path = tmp_path / "policy.pt"
    checkpoint(path, policy())
    backend = DiffusionPolicyBackend({"checkpoint": str(path), "device": "cpu"})
    service = InferenceService(backend, batch_size=2,
        action_expert=dict(kernel="ddim", denoising_steps=5, eta=.5, min_std=.02))
    image = encode_image(np.full((64, 64, 3), 100, dtype=np.uint8))
    row = {"state": [.1, .2], "task": "Press the red button.", "images": {"up": image, "wrist": image}}
    response = service.action_expert.sample_chain(dict(observations=[row], seeds=[17], expected_policy_version=0))
    sample = response["items"][0]
    chain = tree_from_wire(sample["chain"])[None]
    old = tree_from_wire(sample["logprobs"])[None]
    context = tree_from_wire(sample["conditioning"])
    expert = service.action_expert.expert
    assert all(p.requires_grad for p in expert.parameters())
    assert not any(p.requires_grad for p in backend.model.encoder.parameters())
    scorer = GaussianChainPolicy(expert)
    new = torch.stack([scorer.log_prob(context, chain[:, k], chain[:, k+1], k) for k in range(5)], dim=1)
    torch.testing.assert_close(new, old)
    (-new.mean()).backward()
    assert any(p.grad is not None for p in expert.parameters())
    assert all(p.grad is None for p in backend.model.encoder.parameters())
    service.close()


def test_transition_ignores_outer_autocast_without_disabling_gradients():
    model = policy()
    expert = model.action_expert(dict(kernel="ddim", denoising_steps=5, eta=0, min_std=.02))
    x, features = torch.randn(2, 8, 2), torch.randn(2, model.feature_dim)
    expected, _ = expert.transition(x, 2, {"features": features})
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual, std = expert.transition(x, 2, {"features": features})
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == std.dtype == torch.float32
    assert expert.describe()["transition_forward_precision"] == "ieee_float32"
    actual.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in expert.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA batch-kernel parity regression")
def test_cuda_rollout_and_mixed_step_ppo_batch_likelihood_agree():
    """Exercise production-sized UNet B8 rollout versus B256 PPO scoring."""
    from pressb.runtime.denoising import GaussianChainPolicy

    config = configuration()
    config.update(horizon=16, diffusion_steps=100, inference_steps=10,
                  unet_dim=64, unet_dim_mults=[1, 2, 4], time_embedding_dim=64,
                  keypoints=32, task_embedding_dim=32,
                  action_names=[f"j{i}" for i in range(7)],
                  state_names=[f"j{i}" for i in range(7)])
    normalizer = dict(state_center=[0.] * 7, state_scale=[1.] * 7,
                      action_center=[0.] * 7, action_scale=[1.] * 7)
    torch.manual_seed(101)
    model = VisualDiffusionPolicy(config, normalizer, device="cuda")
    expert = model.action_expert(dict(kernel="ddim", denoising_steps=10, eta=0, min_std=.02))
    chain_policy = GaussianChainPolicy(expert)
    features = torch.randn(8, model.feature_dim, device="cuda")
    original = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
    try:
        sample = chain_policy.sample({"features": features}, range(8))
        index = torch.arange(256, device="cuda") % 80
        env, step = index // 10, index % 10
        chain = sample["chain"]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            new = chain_policy.log_prob({"features": features[env]}, chain[env, step],
                                        chain[env, step + 1], step)
        diff = (new - sample["logprobs"][env, step]).detach()
        assert diff.abs().max() < .001
        assert diff.mean((1, 2)).abs().max() < .0001
        (-new.mean()).backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in expert.parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in expert.parameters())
        assert torch.backends.cudnn.allow_tf32 and torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = original
