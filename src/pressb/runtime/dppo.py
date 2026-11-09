"""On-policy denoising PPO mathematics, independent of model and environment.

``upstream_mean`` reproduces DPPO's clipped elementwise-log-probability
heuristic. ``exact_sum`` uses a joint Gaussian transition likelihood instead.
Neither option treats transitions in a replay buffer as on-policy samples.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass
class DPPOConfig:
    rollout_steps: int = 32
    epochs: int = 4
    minibatch_size: int = 256
    actor_lr: float = 1e-5
    critic_lr: float = 3e-4
    weight_decay: float = 0.
    gae_lambda: float = .95
    bootstrap_truncated: bool = True
    gamma_denoising: float = .99
    ft_denoising_steps: int | None = None
    logprob_reduction: str = "upstream_mean"
    logprob_min: float = -5.
    logprob_max: float = 2.
    clip_epsilon: float = .01
    clip_epsilon_min: float = .001
    clip_epsilon_rate: float = 3.
    value_clip: float | None = .2
    value_coefficient: float = .5
    normalize_advantages: bool = True
    max_grad_norm: float = 1.
    target_kl: float | None = .05
    critic_hidden_dim: int = 256
    # Count completed rollout/update iterations, not optimizer minibatches.
    critic_warmup_iterations: int = 0
    # Preserve historical KL early-stop behavior unless explicitly enabled.
    critic_continue_after_kl: bool = False
    # Log probabilities must match the sampling precision before PPO starts.
    initial_logprob_atol: float = .05
    initial_reduced_logprob_atol: float = .0005

    def __post_init__(self):
        for key in ("rollout_steps", "epochs", "minibatch_size", "critic_hidden_dim"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"dppo.{key} must be a positive integer")
        if type(self.critic_warmup_iterations) is not int or self.critic_warmup_iterations < 0:
            raise ValueError("dppo.critic_warmup_iterations must be a nonnegative integer")
        if self.ft_denoising_steps is not None and (type(self.ft_denoising_steps) is not int
                                                   or self.ft_denoising_steps < 1):
            raise ValueError("ft_denoising_steps must be positive or null")
        for key in ("actor_lr", "critic_lr", "max_grad_norm", "initial_logprob_atol", "initial_reduced_logprob_atol"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"dppo.{key} must be finite and positive")
        for key in ("gae_lambda", "gamma_denoising"):
            if type(getattr(self, key)) not in (int, float) or not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"dppo.{key} must be in [0,1]")
        for key in ("bootstrap_truncated", "normalize_advantages", "critic_continue_after_kl"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"dppo.{key} must be boolean")
        for key in ("weight_decay", "value_coefficient", "clip_epsilon", "clip_epsilon_min", "clip_epsilon_rate"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"dppo.{key} must be finite and nonnegative")
        if not 0 < self.clip_epsilon_min <= self.clip_epsilon < 1:
            raise ValueError("Require 0 < clip_epsilon_min <= clip_epsilon < 1")
        for key in ("value_clip", "target_kl"):
            value = getattr(self, key)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
                raise ValueError(f"dppo.{key} must be positive or null")
        if self.logprob_reduction not in ("upstream_mean", "exact_sum"):
            raise ValueError("logprob_reduction must be upstream_mean or exact_sum")
        if not math.isfinite(self.logprob_min) or not math.isfinite(self.logprob_max) or self.logprob_min >= self.logprob_max:
            raise ValueError("Require finite logprob_min < logprob_max")


def generalized_advantages(rewards, values, next_values, discounts, terminated, truncated,
                           *, gae_lambda=.95, bootstrap_truncated=True):
    """GAE for aligned [time, environment] tensors and actual chunk discounts.

    Time limits optionally bootstrap the *terminal observation*, never the reset
    observation. Both termination flags cut the recursion across episode resets.
    The final rollout boundary bootstraps ``next_values[-1]`` without inventing a
    terminal flag. No extra gamma is multiplied into the simulator discount.
    """
    arrays = (rewards, values, next_values, discounts, terminated, truncated)
    if rewards.ndim not in (1, 2) or any(item.shape != rewards.shape for item in arrays):
        raise ValueError("GAE inputs must share [time] or [time,environment] shape")
    if not 0 <= gae_lambda <= 1 or type(bootstrap_truncated) is not bool:
        raise ValueError("Invalid GAE settings")
    if terminated.dtype != torch.bool or truncated.dtype != torch.bool or torch.any(terminated & truncated):
        raise ValueError("Exclusive boolean termination flags required")
    if any(not torch.isfinite(item).all() for item in arrays[:4]) or torch.any((discounts < 0) | (discounts > 1)):
        raise ValueError("GAE needs finite values and discounts in [0,1]")
    advantages = torch.zeros_like(values)
    following = torch.zeros_like(values[0])
    for index in reversed(range(len(values))):
        bootstrap = ~terminated[index]
        if not bootstrap_truncated:
            bootstrap = bootstrap & ~truncated[index]
        delta = rewards[index] + discounts[index] * bootstrap * next_values[index] - values[index]
        continue_episode = ~(terminated[index] | truncated[index])
        following = delta + discounts[index] * gae_lambda * continue_episode * following
        advantages[index] = following
    return advantages, advantages + values


def reduce_logprobs(logprobs, config: DPPOConfig, *, execution_horizon=None):
    """Reduce a denoising transition [batch,horizon,latent_action] to [batch]."""
    if logprobs.ndim != 3 or not torch.isfinite(logprobs).all():
        raise ValueError("Log probabilities must be finite [batch,horizon,action]")
    if execution_horizon is not None:
        if not 1 <= execution_horizon <= logprobs.shape[1]:
            raise ValueError("Invalid log-probability action prefix")
        # A denoiser mixes all coordinates across subsequent transitions. The
        # complete conditional density therefore includes unexecuted positions.
        if config.logprob_reduction == "upstream_mean":
            logprobs = logprobs[:, :execution_horizon]
    logprobs = logprobs.float()
    if config.logprob_reduction == "upstream_mean":
        return logprobs.clamp(config.logprob_min, config.logprob_max).mean(dim=(-2, -1))
    return logprobs.sum(dim=(-2, -1))


def denoising_clip(steps, total_steps, config: DPPOConfig):
    if total_steps < 1 or torch.any((steps < 0) | (steps >= total_steps)):
        raise ValueError("Invalid fine-tuning denoising index")
    # One-step chains use the final, largest clip; avoid upstream's K-1 divide.
    fraction = torch.ones_like(steps, dtype=torch.float32) if total_steps == 1 else steps.float() / (total_steps - 1)
    rate = config.clip_epsilon_rate
    schedule = fraction if rate == 0 else torch.expm1(rate * fraction) / math.expm1(rate)
    return config.clip_epsilon_min + (config.clip_epsilon - config.clip_epsilon_min) * schedule


def clipped_policy_loss(new_logprobs, old_logprobs, advantages, steps, total_steps,
                        config: DPPOConfig, *, execution_horizon=None):
    current = reduce_logprobs(new_logprobs, config, execution_horizon=execution_horizon)
    previous = reduce_logprobs(old_logprobs, config, execution_horizon=execution_horizon).detach()
    if advantages.shape != current.shape or steps.shape != current.shape:
        raise ValueError("PPO advantages and denoising steps must match batch")
    log_ratio = current - previous
    ratio = log_ratio.exp()
    if not torch.isfinite(ratio).all():
        raise FloatingPointError("Nonfinite importance ratio; reject update rather than silently clipping likelihoods")
    epsilon = denoising_clip(steps, total_steps, config)
    discounted = advantages.detach() * config.gamma_denoising ** (total_steps - steps - 1)
    loss = torch.maximum(-discounted * ratio, -discounted * ratio.clamp(1 - epsilon, 1 + epsilon)).mean()
    metrics = {"policy_loss": float(loss.detach()), "approx_kl": float(((ratio - 1) - log_ratio).mean().detach()),
               "clip_fraction": float(((ratio - 1).abs() > epsilon).float().mean().detach()),
               "mean_ratio": float(ratio.mean().detach())}
    return loss, metrics


def clipped_value_loss(values, old_values, returns, clip=None):
    if values.shape != old_values.shape or values.shape != returns.shape:
        raise ValueError("Value tensors must have identical shapes")
    loss = (values - returns).square()
    if clip is not None:
        clipped = old_values + (values - old_values).clamp(-clip, clip)
        loss = torch.maximum(loss, (clipped - returns).square())
    return .5 * loss.mean()
