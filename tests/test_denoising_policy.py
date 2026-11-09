"""Behavioral tests for the distribution sampled and optimized by chain PPO."""
import math

import pytest
import torch
from torch import nn

from pressb.runtime.denoising import GaussianChainPolicy


class LinearExpert(nn.Module):
    horizon = 3
    latent_dim = 2
    denoising_steps = 4

    def __init__(self, std=0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.3))
        self.std = std
        self.dropout = nn.Dropout(0.8)

    @property
    def device(self):
        return self.weight.device

    def transition(self, x, steps, condition):
        mean = self.dropout(x) * self.weight + condition[:, None, None] + steps[:, None, None] * 0.01
        return mean, torch.as_tensor(self.std, device=x.device)

    def actions_from_latent(self, x):
        return x.clamp_(-0.25, 0.25)


def test_sample_density_recomputes_exactly_and_preserves_unclipped_latents():
    policy = GaussianChainPolicy(LinearExpert())
    result = policy.sample(torch.tensor([0.8, -0.8]), [7, 11])
    assert result["chain"].shape == (2, 5, 3, 2)
    assert result["logprobs"].shape == (2, 4, 3, 2)
    assert result["actions"].shape == (2, 3, 2)
    assert result["chain"][:, -1].abs().max() > 0.25
    assert result["actions"].abs().max() <= 0.25
    for step in range(4):
        recomputed = policy.log_prob(torch.tensor([0.8, -0.8]),
                                     result["chain"][:, step], result["chain"][:, step + 1], step)
        torch.testing.assert_close(recomputed, result["logprobs"][:, step], rtol=0, atol=0)


def test_per_row_seed_reordering_microbatching_and_global_rng_independence():
    policy = GaussianChainPolicy(LinearExpert())
    torch.manual_seed(101)
    rng = torch.get_rng_state().clone()
    batch = policy.sample(torch.tensor([0.2, 0.4, 0.1]), [9, 23, 31])
    assert torch.equal(torch.get_rng_state(), rng)
    torch.randn(100)
    for row, (condition, seed) in enumerate(zip([0.2, 0.4, 0.1], [9, 23, 31])):
        single = policy.sample(torch.tensor([condition]), [seed])
        torch.testing.assert_close(single["chain"][0], batch["chain"][row], rtol=0, atol=0)
    reverse = policy.sample(torch.tensor([0.1, 0.4, 0.2]), [31, 23, 9])
    torch.testing.assert_close(reverse["chain"], batch["chain"].flip(0), rtol=0, atol=0)


def test_explicit_initial_noise_preserves_rng_for_subsequent_transitions():
    policy = GaussianChainPolicy(LinearExpert())
    result = policy.sample(torch.tensor([0.2]), [1])
    noise = result["chain"][:, 0].clone()
    again = policy.sample(torch.tensor([0.2]), [1], initial_noise=noise)
    torch.testing.assert_close(result["chain"], again["chain"], rtol=0, atol=0)
    assert torch.equal(noise, result["chain"][:, 0])


def test_deterministic_evaluation_is_not_mislabeled_as_stochastic_data():
    policy = GaussianChainPolicy(LinearExpert(std=0))
    result = policy.sample(torch.tensor([0.1]), [4], deterministic=True)
    assert result["logprobs"] is None
    for step in range(4):
        mean, _ = policy.expert.transition(result["chain"][:, step], torch.tensor([step]), torch.tensor([0.1]))
        torch.testing.assert_close(result["chain"][:, step + 1], mean)
    with pytest.raises(ValueError, match="strictly positive"):
        policy.sample(torch.tensor([0.1]), [4])
    with pytest.raises(ValueError, match="strictly positive"):
        policy.log_prob(torch.tensor([0.1]), result["chain"][:, 0], result["chain"][:, 1], 0)


def test_expert_gradient_matches_gaussian_derivative_without_pathwise_latent_gradients():
    expert = LinearExpert()
    policy = GaussianChainPolicy(expert)
    x = torch.ones(1, 3, 2, requires_grad=True)
    next_x = torch.full((1, 3, 2), 0.7, requires_grad=True)
    log_prob = policy.log_prob(torch.tensor([0.1]), x, next_x, torch.tensor([0]))
    expected = -0.5 * ((0.7 - 0.4) / 0.2) ** 2 - math.log(0.2) - 0.5 * math.log(2 * math.pi)
    torch.testing.assert_close(log_prob, torch.full_like(log_prob, expected))
    (-log_prob.mean()).backward()
    assert expert.weight.grad.item() == pytest.approx(-(0.7 - 0.4) / 0.2**2)
    assert x.grad is None and next_x.grad is None
    before = expert.weight.detach().clone()
    torch.optim.SGD(expert.parameters(), lr=0.01).step()
    assert not torch.equal(expert.weight, before)


def test_network_dropout_disabled_even_if_caller_sets_train_mode():
    expert = LinearExpert()
    policy = GaussianChainPolicy(expert)
    a = policy.sample(torch.tensor([0.1]), [13])
    expert.train()
    b = policy.sample(torch.tensor([0.1]), [13])
    assert not expert.dropout.training
    torch.testing.assert_close(a["chain"], b["chain"], rtol=0, atol=0)
    expert.train()
    scored = policy.log_prob(torch.tensor([0.1]), a["chain"][:, 0], a["chain"][:, 1], 0)
    torch.testing.assert_close(scored, a["logprobs"][:, 0], rtol=0, atol=0)


def test_bfloat16_expert_output_scored_in_float32():
    class BF16Expert(LinearExpert):
        def transition(self, x, steps, condition):
            mean, std = super().transition(x, steps, condition)
            return mean.to(torch.bfloat16), std.to(torch.bfloat16)

    policy = GaussianChainPolicy(BF16Expert())
    sampled = policy.sample(torch.tensor([0.1]), [2])
    assert all(value.dtype == torch.float32 for value in sampled.values())
    lp = policy.log_prob(torch.tensor([0.1]), sampled["chain"][:, 0], sampled["chain"][:, 1], 0)
    torch.testing.assert_close(lp, sampled["logprobs"][:, 0], rtol=0, atol=0)
    assert lp.requires_grad


@pytest.mark.parametrize("std", [-0.1, float("nan"), float("inf"), torch.ones(2, 5)])
def test_rejects_invalid_transition_variances(std):
    policy = GaussianChainPolicy(LinearExpert(std=std))
    with pytest.raises(ValueError, match="std"):
        policy.sample(torch.tensor([0.1]), [1])


@pytest.mark.parametrize("seeds", [[], [True], [-1], [1.5], [2**63]])
def test_rejects_invalid_seeds(seeds):
    with pytest.raises(ValueError, match="seeds"):
        GaussianChainPolicy(LinearExpert()).sample(torch.tensor([0.1]), seeds)


@pytest.mark.parametrize("steps", [-1, 4, 1.1, True, [0, 1]])
def test_rejects_invalid_steps(steps):
    with pytest.raises(ValueError, match="steps"):
        GaussianChainPolicy(LinearExpert()).log_prob(torch.tensor([0.1]), torch.ones(1, 3, 2), torch.ones(1, 3, 2), steps)


@pytest.mark.parametrize("noise", [torch.zeros(1, 2, 2), torch.zeros(2, 3, 2),
                                  torch.ones(1, 3, 2, dtype=torch.long), torch.full((1, 3, 2), float("nan"))])
def test_rejects_invalid_initial_noise(noise):
    with pytest.raises(ValueError, match="initial_noise"):
        GaussianChainPolicy(LinearExpert()).sample(torch.tensor([0.1]), [1], initial_noise=noise)


def test_gaussian_density_has_correct_empirical_moments():
    expert = LinearExpert()
    expert.denoising_steps = 1
    policy = GaussianChainPolicy(expert)
    sampled = policy.sample(torch.full((5000,), 0.7), range(5000), initial_noise=torch.zeros(5000, 3, 2))
    residual = sampled["chain"][:, 1].reshape(-1) - 0.7
    assert abs(residual.mean().item()) < 0.005
    assert residual.std().item() == pytest.approx(0.2, abs=0.003)
