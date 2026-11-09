"""Gaussian denoising chains shared by diffusion and stochastic-flow PPO.

The expert supplies a conditional Gaussian *transition*, not an action-space
likelihood. Flow experts must explicitly add stochastic transitions; the
original deterministic ODE has no Gaussian density. Latents and log densities
are kept in float32 even when the expert network runs in bfloat16.

This module deliberately does not reduce, temper, or clip log probabilities.
DPPO's upstream mean/clamp surrogate and an exact joint-density sum are choices
for the PPO objective, not properties of the sampling distribution.
"""
from __future__ import annotations

import math
from numbers import Integral
from typing import Any

import torch
from torch import nn


class GaussianChainPolicy:
    """Sample and score the same finite Gaussian Markov chain.

    An expert implements ``transition(x, steps, condition) -> (mean, std)`` and
    ``actions_from_latent(x)``. It exposes positive integer ``horizon``,
    ``latent_dim``, ``denoising_steps`` and a ``device`` (or parameters). Step
    indices run from 0 at initial noise to K-1 before the final action latent;
    mapping those indices to DDPM/DDIM timesteps belongs to the expert.

    Sampling stores the unmodified latent before action conversion, clipping or
    smoothing. Every stochastic transition, including the final one, must have
    positive variance. Deterministic evaluation uses means and returns no log
    probabilities: its zero-variance transitions must never enter PPO batches.
    """

    def __init__(self, expert: Any):
        self.expert = expert
        for name in ("horizon", "latent_dim", "denoising_steps"):
            value = getattr(expert, name, None)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"expert.{name} must be a positive integer")
            setattr(self, name, int(value))
        for name in ("transition", "actions_from_latent"):
            if not callable(getattr(expert, name, None)):
                raise TypeError(f"expert must implement {name}")
        self._eval_networks()

    @property
    def device(self) -> torch.device:
        device = getattr(self.expert, "device", None)
        if device is not None:
            return torch.device(device)
        parameters = getattr(self.expert, "parameters", None)
        if callable(parameters):
            first = next(iter(parameters()), None)
            if first is not None:
                return first.device
        return torch.device("cpu")

    def _eval_networks(self) -> None:
        # eval() disables dropout/BN updates but does not disable autograd. A
        # lightweight adapter may own a Module without itself inheriting it.
        evaluator = getattr(self.expert, "eval", None)
        if callable(evaluator):
            evaluator()
        for value in vars(self.expert).values() if hasattr(self.expert, "__dict__") else ():
            if isinstance(value, nn.Module):
                value.eval()

    def _latent(self, value: Any, name: str, batch: int | None = None) -> torch.Tensor:
        value = torch.as_tensor(value, device=self.device)
        if not value.is_floating_point():
            raise ValueError(f"{name} must be a floating point tensor")
        if value.ndim != 3 or tuple(value.shape[1:]) != (self.horizon, self.latent_dim):
            raise ValueError(f"{name} must have shape [B, {self.horizon}, {self.latent_dim}]")
        if value.shape[0] < 1 or (batch is not None and value.shape[0] != batch):
            raise ValueError(f"{name} batch size mismatch")
        value = value.to(dtype=torch.float32)
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain finite float32 values")
        return value

    def _steps(self, steps: Any, batch: int) -> torch.Tensor:
        steps = torch.as_tensor(steps, device=self.device)
        if steps.dtype == torch.bool or steps.is_floating_point() or steps.is_complex():
            raise ValueError("steps must contain integer denoising indices")
        if steps.ndim == 0:
            steps = steps.expand(batch)
        if tuple(steps.shape) != (batch,):
            raise ValueError("steps must be scalar or have shape [B]")
        steps = steps.to(dtype=torch.long)
        if (steps < 0).any() or (steps >= self.denoising_steps).any():
            raise ValueError("steps are outside the expert denoising schedule")
        return steps

    def _transition(self, condition: Any, x: torch.Tensor, steps: torch.Tensor,
                    *, require_density: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std = self.expert.transition(x, steps, condition)
        mean = self._latent(mean, "transition mean", x.shape[0])
        std = torch.as_tensor(std, dtype=torch.float32, device=self.device)
        try:
            std = torch.broadcast_to(std, mean.shape)
        except RuntimeError as exc:
            raise ValueError("transition std must broadcast to [B, H, D]") from exc
        if not torch.isfinite(std).all() or (std < 0).any():
            raise ValueError("transition std must be finite and nonnegative")
        if require_density and (std == 0).any():
            raise ValueError("stochastic transitions require strictly positive std")
        return mean, std

    @staticmethod
    def _gaussian_log_prob(mean: torch.Tensor, std: torch.Tensor,
                           next_x: torch.Tensor) -> torch.Tensor:
        log_prob = -0.5 * ((next_x - mean) / std).square() - std.log() - 0.5 * math.log(2 * math.pi)
        if not torch.isfinite(log_prob).all():
            raise ValueError("Gaussian chain log probabilities are nonfinite")
        return log_prob

    def log_prob(self, condition: Any, x: Any, next_x: Any, steps: Any) -> torch.Tensor:
        """Differentiable, elementwise log density at fixed sampled transitions.

        The two rollout latents are detached: PPO differentiates the transition
        model at recorded states, not a newly unrolled denoising trajectory.
        """
        self._eval_networks()
        x = self._latent(x, "x").detach()
        next_x = self._latent(next_x, "next_x", x.shape[0]).detach()
        steps = self._steps(steps, x.shape[0])
        mean, std = self._transition(condition, x, steps)
        return self._gaussian_log_prob(mean, std, next_x)

    @torch.no_grad()
    def sample(self, condition: Any, seeds: Any, deterministic: bool = False,
               initial_noise: Any = None) -> dict[str, torch.Tensor | None]:
        """Return actions, the raw ``[B,K+1,H,D]`` chain, and raw log densities.

        Every row owns its RNG. Its noise draws are independent of microbatch
        size, ordering, and unrelated global RNG use. Network kernels may still
        have normal floating-point differences between batch shapes.
        """
        if type(deterministic) is not bool:
            raise ValueError("deterministic must be a boolean")
        seeds = list(seeds)
        if not seeds or any(isinstance(seed, bool) or not isinstance(seed, Integral)
                            or seed < 0 or seed >= 2**63 for seed in seeds):
            raise ValueError("seeds must be a nonempty sequence of integers in [0, 2**63)")
        self._eval_networks()
        generators = [torch.Generator(device=self.device).manual_seed(int(seed)) for seed in seeds]

        def draw():
            return torch.stack([torch.randn((self.horizon, self.latent_dim),
                                generator=generator, device=self.device, dtype=torch.float32)
                                for generator in generators])

        # Consume the same initial RNG draw even for caller-specified noise, so
        # changing only initial_noise does not also change subsequent epsilons.
        noise = draw()
        x = noise if initial_noise is None else self._latent(initial_noise, "initial_noise", len(seeds)).clone()
        chain = [x.clone()]
        log_probs = []
        for step in range(self.denoising_steps):
            steps = torch.full((len(seeds),), step, dtype=torch.long, device=self.device)
            mean, std = self._transition(condition, x, steps, require_density=not deterministic)
            next_x = mean if deterministic else mean + std * draw()
            if not torch.isfinite(next_x).all():
                raise ValueError("sampled latent contains nonfinite values")
            if not deterministic:
                log_probs.append(self._gaussian_log_prob(mean, std, next_x))
            chain.append(next_x.clone())
            x = next_x
        # Give converters their own tensor: an in-place action clamp must not
        # rewrite the sampled chain and invalidate its density.
        actions = torch.as_tensor(self.expert.actions_from_latent(x.clone()), device=self.device)
        if (actions.ndim != 3 or actions.shape[:2] != (len(seeds), self.horizon)
                or actions.shape[-1] < 1 or not actions.is_floating_point()
                or not torch.isfinite(actions).all()):
            raise ValueError("actions_from_latent must return finite floating point [B,H,A]")
        actions = actions.to(dtype=torch.float32)
        if not torch.isfinite(actions).all():
            raise ValueError("converted actions overflow float32")
        return {"chain": torch.stack(chain, dim=1),
                "logprobs": None if deterministic else torch.stack(log_probs, dim=1),
                "actions": actions}
