"""CPU checks for algorithm boundaries and exact learner/replay continuation."""
from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from pressb.online_rl.learner import LearnerConfig, ReplayBuffer, SACLearner


@pytest.fixture(autouse=True, scope="module")
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_learner(method="action_residual", **kwargs):
    return SACLearner(LearnerConfig(method=method, obs_dim=4, hidden_dim=8,
                                   batch_size=4, **kwargs), device="cpu")


def make_batch(learner, count=4):
    rng = np.random.default_rng(51)
    batch = {"obs": rng.normal(size=(count, 4)).astype(np.float32),
             "next_obs": rng.normal(size=(count, 4)).astype(np.float32),
             "action": rng.uniform(-1, 1, size=(count, learner.action_dim)).astype(np.float32),
             "reward": np.zeros((count, 1), np.float32),
             "discount": np.full((count, 1), .93, np.float32),
             "terminated": np.zeros((count, 1), np.bool_),
             "truncated": np.zeros((count, 1), np.bool_)}
    if learner.residual:
        batch["base_action"] = rng.normal(size=(count, 7, 9)).astype(np.float32)
        batch["next_base_action"] = rng.normal(size=(count, 7, 9)).astype(np.float32)
    return batch


class ConstantQs(nn.Module):
    def __init__(self, count, value):
        super().__init__()
        self.count, self.value = count, value
        self.last_action = None

    def forward(self, obs, action):
        self.last_action = action.detach().clone()
        return action.sum(-1, keepdim=True).unsqueeze(0).expand(self.count, -1, -1) * 0 + self.value


@pytest.mark.parametrize("method,qs,freq,dim", [
    ("action_residual", 2, 2, 63), ("initial_noise", 5, 1, 9)])
def test_method_defaults_and_network_contract(method, qs, freq, dim):
    learner = make_learner(method)
    assert learner.action_dim == dim
    assert learner.config.num_qs == qs and learner.config.target_freq == freq
    assert learner.config.lambda_s is None and learner.config.lambda_t is None
    assert not learner.config.bootstrap_truncated
    assert all(not p.requires_grad for p in learner.q_targets.parameters())
    assert any(isinstance(m, nn.LayerNorm) for m in learner.actor.modules()) == (method == "initial_noise")
    actions = torch.zeros(3, dim)
    with torch.no_grad():
        learner.qs.net[-1].weight.zero_()
        learner.qs.net[-1].bias.fill_(3)
        values = learner.qs(torch.zeros(3, 4), actions)
    if method == "action_residual":
        assert (values >= 0).all() and (values <= 1).all()
    else:
        torch.testing.assert_close(values, torch.full_like(values, 3))


@pytest.mark.parametrize("override", [dict(noise_steps=2), dict(num_subset=6),
    dict(residual_scale=[.1] * 8), dict(tau=0), dict(lambda_s=.1), dict(sigma=-1)])
def test_invalid_algorithm_contract_is_rejected(override):
    with pytest.raises(ValueError):
        LearnerConfig(method="action_residual", obs_dim=4, **override)


def test_residual_warmup_and_progressive_exploration_execute_whole_chunks():
    learner = make_learner()
    obs, base = np.zeros((128, 4), np.float32), np.zeros((128, 7, 9), np.float32)
    assert np.count_nonzero(learner.act(obs, base, warmup=True)) == 0
    with torch.no_grad():
        learner.actor.mean.weight.zero_()
        learner.actor.mean.bias.fill_(.7)
    action = learner.act(obs, base, deterministic=True, exploration_probability=.5)
    zero_rows = np.all(action == 0, axis=1)
    assert 0 < zero_rows.sum() < len(action)
    assert np.all(action[~zero_rows] > 0)
    assert not np.any(np.any(action == 0, axis=1) & ~zero_rows)
    np.testing.assert_array_equal(learner.act(obs, base, exploration_probability=0), np.zeros_like(action))
    first = learner.act(obs, base, deterministic=True)
    second = learner.act(obs, base, deterministic=True)
    np.testing.assert_array_equal(first, second)


def test_noise_warmup_returns_the_actual_clipped_normalized_noise():
    learner = make_learner("initial_noise", noise_steps=7)
    draw = torch.linspace(-3., 3., 63).reshape(1, 63)
    with patch("pressb.online_rl.learner.torch.randn", return_value=draw):
        action = learner.act(np.zeros((1, 4), np.float32), warmup=True)
    expected_noise = np.clip(draw.numpy(), -1.5, 1.5)
    np.testing.assert_array_equal(action * 1.5, expected_noise)
    assert action.shape == (1, 63)
    np.testing.assert_array_equal(learner.act(np.zeros((1, 4)), deterministic=True),
                                  learner.act(np.zeros((1, 4)), deterministic=True))


@pytest.mark.parametrize("method", ["action_residual", "initial_noise"])
@pytest.mark.parametrize("bootstrap", [False, True])
def test_variable_duration_discount_and_terminal_bootstrap(method, bootstrap):
    learner = make_learner(method, bootstrap_truncated=bootstrap, auto_alpha=False,
                           init_alpha=.25, num_qs=2, num_subset=2)
    batch = make_batch(learner)
    batch["reward"][:, 0] = [.1, 1., 0., 0.]
    batch["discount"][:, 0] = [.5, .2, .8, .99]
    batch["terminated"][1] = True
    batch["truncated"][2] = True
    learner.q_targets = ConstantQs(2, .5)

    def actor_sample(obs):
        zeros = torch.zeros(len(obs), learner.action_dim)
        return zeros, torch.full((len(obs), 1), -2.), zeros

    with patch.object(learner.actor, "sample", side_effect=actor_sample):
        target = learner._bellman_target(learner._prepare_batch(batch)).numpy().flatten()
    # Noise's soft target is .5 - .25*(-2)=1; residual omits entropy.
    next_value = .5 if method == "action_residual" else 1.
    expected = [.1 + .5 * next_value, 1., .8 * next_value if bootstrap else 0., .99 * next_value]
    np.testing.assert_allclose(target, expected, rtol=1e-6)
    if learner.residual:
        np.testing.assert_array_equal(learner.q_targets.last_action.numpy(), batch["next_base_action"].reshape(4, 63))


def test_noise_entropy_switch_only_changes_target():
    learner = make_learner("initial_noise", q_entropy=False, auto_alpha=False, init_alpha=.25)
    learner.q_targets = ConstantQs(5, .5)
    batch = learner._prepare_batch(make_batch(learner))
    with patch.object(learner.actor, "sample", return_value=(torch.zeros(4, 9), torch.full((4, 1), -2.), torch.zeros(4, 9))):
        hard = learner._bellman_target(batch)
        learner.config.q_entropy = True
        soft = learner._bellman_target(batch)
    torch.testing.assert_close(soft - hard, batch["discount"] * .5)


@pytest.mark.parametrize("method", ["action_residual", "initial_noise"])
def test_updates_polyak_and_frozen_targets(method):
    learner = make_learner(method, policy_freq=2, target_freq=2, tau=.2)
    batch = make_batch(learner)
    before_actor = [p.detach().clone() for p in learner.actor.parameters()]
    initial_target = [p.detach().clone() for p in learner.q_targets.parameters()]
    first = learner.update(batch)
    assert first["actor_updated"] == first["target_updated"] == 0
    for before, after in zip(before_actor, learner.actor.parameters()):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    for before, after in zip(initial_target, learner.q_targets.parameters()):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    second = learner.update(batch)
    assert second["actor_updated"] == second["target_updated"] == 1
    assert all(np.isfinite(v) for v in second.values())
    for before, current, after in zip(initial_target, learner.qs.parameters(), learner.q_targets.parameters()):
        torch.testing.assert_close(after, .8 * before + .2 * current)
        assert after.grad is None and not after.requires_grad
    assert any(not torch.equal(before, after) for before, after in zip(before_actor, learner.actor.parameters()))
    assert learner.alpha_optimizer.state and learner.actor_optimizer.state and learner.q_optimizer.state


def test_caps_penalizes_final_pose9_chunk_and_is_optional():
    learner = make_learner(lambda_s=0., lambda_t=2., sigma=.01, auto_alpha=False)
    batch = make_batch(learner)
    ramp = np.arange(7, dtype=np.float32)[None, :, None] * np.ones((4, 7, 9), np.float32)
    batch["base_action"] = ramp
    learner.qs = ConstantQs(2, 0.)
    with torch.no_grad():
        for param in learner.actor.parameters():
            param.zero_()
    torch.manual_seed(1)
    enabled, info = learner._actor_loss(learner._prepare_batch(batch))
    assert info["temporal_smoothness_loss"] == pytest.approx(.5 * 9)
    assert info["spatial_smoothness_loss"] == 0
    assert enabled.item() - info["actor_rl_loss"] == pytest.approx(9., abs=1e-6)
    learner.config.lambda_s = learner.config.lambda_t = None
    torch.manual_seed(1)
    disabled, disabled_info = learner._actor_loss(learner._prepare_batch(batch))
    assert disabled.item() == pytest.approx(disabled_info["actor_rl_loss"])


def test_replay_roundtrip_preserves_wraparound_rng_and_source_copy():
    replay = ReplayBuffer(3, 4, 63, seed=25)
    base = np.zeros((7, 9), np.float32)
    for i in range(5):
        replay.add(np.full(4, i), np.zeros(63), float(i), .9, np.full(4, i + 1),
                   False, bool(i == 4), base, base + 1)
    base.fill(100.)
    assert len(replay) == 3
    assert np.all(replay.arrays["base_action"] == 0)
    snapshot = replay.state_dict()
    restored = ReplayBuffer(3, 4, 63, seed=100)
    restored.load_state_dict(snapshot)
    assert restored.position == replay.position
    expected, actual = replay.sample(10), restored.sample(10)
    for key in expected:
        np.testing.assert_array_equal(expected[key], actual[key])
    snapshot["arrays"]["obs"].fill(-1)
    assert np.all(restored.arrays["obs"] >= 2)
    with pytest.raises(ValueError, match="mix"):
        restored.add(np.zeros(4), np.zeros(63), 0, .9, np.zeros(4), False, False)
    bad = restored.state_dict()
    bad["arrays"]["discount"][0] = 1.5
    with pytest.raises(ValueError, match="discount"):
        restored.load_state_dict(bad)


@pytest.mark.parametrize("method", ["action_residual", "initial_noise"])
def test_checkpoint_continues_rng_optimizer_and_updates_exactly(tmp_path, method):
    learner = make_learner(method, policy_freq=1, target_freq=1)
    batch = make_batch(learner)
    learner.update(batch)
    path = tmp_path / "checkpoint.pt"
    learner.save(path, {"base_sha": "test-fixture", "counter": 42})
    base = batch.get("base_action")
    expected_action = learner.act(batch["obs"], base, exploration_probability=.5)
    expected_metrics = learner.update(batch)
    restored = SACLearner.load(path, device="cpu")
    assert restored.checkpoint_extra == {"base_sha": "test-fixture", "counter": 42}
    np.testing.assert_array_equal(restored.act(batch["obs"], base, exploration_probability=.5), expected_action)
    actual_metrics = restored.update(batch)
    assert actual_metrics == expected_metrics
    assert learner.updates == restored.updates == 2
    for name in ("actor", "qs", "q_targets"):
        for expected, actual in zip(getattr(learner, name).parameters(), getattr(restored, name).parameters()):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    torch.testing.assert_close(learner.log_alpha, restored.log_alpha, rtol=0, atol=0)
    assert all(not p.requires_grad for p in restored.q_targets.parameters())


def test_malformed_transition_does_not_mutate_replay_or_learner():
    replay = ReplayBuffer(2, 4, 9)
    with pytest.raises(ValueError, match="discount"):
        replay.add(np.zeros(4), np.zeros(9), 0, float("nan"), np.zeros(4), False, False)
    assert len(replay) == replay.position == 0
    learner = make_learner("initial_noise")
    batch = make_batch(learner)
    batch["terminated"][0] = batch["truncated"][0] = True
    with pytest.raises(ValueError, match="both terminated"):
        learner.update(batch)
    assert learner.updates == 0
