"""Chunk-level SAC learners adapted from the local ZPRL implementation.

Source: pgq18/ZPRL, commit a34a1cb, zprl/policy/{residue,noise}_policy.py
and zprl/model/online.py. The residual and noise variants intentionally have
different networks, Bellman targets and actor-Q reductions. No frozen VLA
model is imported here. Residual critics see the combined, unprojected pose9.

ZPRL portions and their adaptations are distributed under the MIT License:

Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab
Copyright (c) 2026 Dongjie Yu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


HORIZON = 7
POSE_DIM = 9
SOURCE_COMMIT = "a34a1cb"
CHECKPOINT_VERSION = 2


@dataclass
class LearnerConfig:
    method: str
    obs_dim: int
    hidden_dim: int = 256
    noise_steps: int = 1
    noise_scale: float = 1.5
    residual_scale: tuple[float, ...] = field(
        default_factory=lambda: (.03, .03, .03, .1, .1, .1, .1, .1, .1))
    # xyz has three learned outputs per step; rotations remain in the frozen
    # base action used for actor conditioning and critic evaluation.
    residual_mode: str = "pose9"
    batch_size: int = 256
    policy_lr: float = 1e-4
    q_lr: float = 3e-4
    tau: float = .005
    init_alpha: float = .01
    auto_alpha: bool = True
    # None chooses -action_dim/2 for residual and 0 for initial-noise SAC.
    target_entropy: float | None = None
    num_qs: int | None = None
    num_subset: int = 2
    policy_freq: int = 2
    target_freq: int | None = None
    bootstrap_truncated: bool = False
    q_entropy: bool = True
    log_std_min: float = -10.
    log_std_max: float = 2.
    max_grad_norm: float = 50.
    lambda_s: float | None = None
    lambda_t: float | None = None
    sigma: float = .01
    seed: int = 42

    def __post_init__(self):
        if self.method not in ("action_residual", "initial_noise"):
            raise ValueError("method must be action_residual or initial_noise")
        if self.residual_mode not in ("pose9", "xyz"):
            raise ValueError("residual_mode must be pose9 or xyz")
        if self.residual_mode == "xyz" and self.method != "action_residual":
            raise ValueError("residual_mode=xyz is only defined for action_residual")
        if self.num_qs is None:
            self.num_qs = 2 if self.method == "action_residual" else 5
        if self.target_freq is None:
            self.target_freq = 2 if self.method == "action_residual" else 1
        for name in ("obs_dim", "hidden_dim", "batch_size", "noise_steps", "num_qs",
                     "num_subset", "policy_freq", "target_freq"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if HORIZON % self.noise_steps:
            raise ValueError("noise_steps must divide the seven-step horizon (1 or 7)")
        if self.num_subset > self.num_qs:
            raise ValueError("num_subset cannot exceed num_qs")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        scale_dim = 3 if self.residual_mode == "xyz" else POSE_DIM
        scale_error = f"residual_scale must contain {scale_dim} finite positive numeric scales"
        try:
            scales = tuple(self.residual_scale)
        except TypeError as error:
            raise ValueError(scale_error) from error
        if (len(scales) != scale_dim or any(
                isinstance(x, (bool, np.bool_))
                or not isinstance(x, (int, float, np.integer, np.floating)) for x in scales)):
            raise ValueError(scale_error)
        self.residual_scale = tuple(float(x) for x in scales)
        if not all(math.isfinite(x) and x > 0 for x in self.residual_scale):
            raise ValueError(scale_error)
        for name in ("noise_scale", "policy_lr", "q_lr", "init_alpha", "max_grad_norm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.tau) or not 0 < self.tau <= 1:
            raise ValueError("tau must be in (0,1]")
        if (not all(math.isfinite(v) for v in (self.log_std_min, self.log_std_max))
                or self.log_std_min >= self.log_std_max):
            raise ValueError("log_std bounds must be finite and increasing")
        if self.target_entropy is not None and not math.isfinite(self.target_entropy):
            raise ValueError("target_entropy must be finite")
        if (self.lambda_s is None) != (self.lambda_t is None):
            raise ValueError("lambda_s and lambda_t must both be set or both be None")
        if self.lambda_s is not None:
            if self.method != "action_residual":
                raise ValueError("CAPS is only defined for action_residual")
            if not all(math.isfinite(v) and v >= 0 for v in (self.lambda_s, self.lambda_t)):
                raise ValueError("CAPS weights must be finite and nonnegative")
        if not math.isfinite(self.sigma) or self.sigma < 0:
            raise ValueError("sigma must be finite and nonnegative")

    @property
    def action_dim(self) -> int:
        if self.method == "action_residual":
            return HORIZON * (3 if self.residual_mode == "xyz" else POSE_DIM)
        return POSE_DIM * self.noise_steps


def _array(value, shape, name, dtype=np.float32):
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {result.shape}")
    return result


class ReplayBuffer:
    """Uniform replay; each row represents one executed action chunk.

    ``discount`` is the environment's actual-duration discount, not a fixed
    gamma. Capacity counts individual transitions, regardless of vector size.
    ``add`` takes a single transition; base actions accept (7,9) or (63,).
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, seed: int = 0):
        if any(type(x) is not int or x < 1 for x in (capacity, obs_dim, action_dim)):
            raise ValueError("Replay dimensions and capacity must be positive integers")
        self.capacity, self.obs_dim, self.action_dim = capacity, obs_dim, action_dim
        self.rng = np.random.default_rng(seed)
        self.position = 0
        self.size = 0
        self._has_base: bool | None = None
        self.arrays = {
            "obs": np.empty((capacity, obs_dim), np.float32),
            "next_obs": np.empty((capacity, obs_dim), np.float32),
            "action": np.empty((capacity, action_dim), np.float32),
            "reward": np.empty((capacity, 1), np.float32),
            "discount": np.empty((capacity, 1), np.float32),
            "terminated": np.empty((capacity, 1), np.bool_),
            "truncated": np.empty((capacity, 1), np.bool_),
        }

    def __len__(self):
        return self.size

    def add(self, obs, action, reward, discount, next_obs, terminated, truncated,
            base_action=None, next_base_action=None):
        if (base_action is None) != (next_base_action is None):
            raise ValueError("Both current and next base actions are required together")
        has_base = base_action is not None
        if self._has_base is not None and has_base != self._has_base:
            raise ValueError("Cannot mix residual and noise transitions in one replay")
        if (not isinstance(terminated, (bool, np.bool_))
                or not isinstance(truncated, (bool, np.bool_)) or (terminated and truncated)):
            raise ValueError("terminated/truncated must be exclusive booleans")
        values = {"obs": _array(obs, (self.obs_dim,), "obs"),
                  "next_obs": _array(next_obs, (self.obs_dim,), "next_obs"),
                  "action": _array(action, (self.action_dim,), "action"),
                  "reward": _array([reward], (1,), "reward"),
                  "discount": _array([discount], (1,), "discount"),
                  "terminated": [terminated], "truncated": [truncated]}
        if not 0 <= float(discount) <= 1:
            raise ValueError("discount must be in [0,1]")
        if np.abs(values["action"]).max() > 1.000001:
            raise ValueError("Replay action must be the executed normalized RL action")
        if has_base:
            for name, value in (("base_action", base_action), ("next_base_action", next_base_action)):
                arr = np.asarray(value)
                if arr.shape == (HORIZON, POSE_DIM):
                    arr = arr.reshape(-1)
                values[name] = _array(arr, (HORIZON * POSE_DIM,), name)
        # Validate everything before mutating the ring or choosing its method.
        if self._has_base is None:
            self._has_base = has_base
            if has_base:
                for name in ("base_action", "next_base_action"):
                    self.arrays[name] = np.empty((self.capacity, HORIZON * POSE_DIM), np.float32)
        for name, value in values.items():
            self.arrays[name][self.position] = value
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if type(batch_size) is not int or batch_size < 1 or self.size == 0:
            raise ValueError("Cannot sample empty replay or a nonpositive batch")
        indices = self.rng.integers(0, self.size, size=batch_size)
        return {name: values[indices].copy() for name, values in self.arrays.items()}

    def state_dict(self) -> dict[str, Any]:
        return {"version": 1, "capacity": self.capacity, "obs_dim": self.obs_dim,
                "action_dim": self.action_dim, "position": self.position, "size": self.size,
                "has_base": self._has_base, "rng": copy.deepcopy(self.rng.bit_generator.state),
                "arrays": {name: value[:self.size].copy() for name, value in self.arrays.items()}}

    def load_state_dict(self, state: dict):
        if (state.get("version") != 1 or state.get("capacity") != self.capacity
                or state.get("obs_dim") != self.obs_dim or state.get("action_dim") != self.action_dim):
            raise ValueError("Replay checkpoint dimensions or version differ")
        size, position, has_base = state["size"], state["position"], state["has_base"]
        if (type(size) is not int or not 0 <= size <= self.capacity
                or type(position) is not int or not 0 <= position < self.capacity
                or (size < self.capacity and position != size)
                or (has_base is not None and type(has_base) is not bool)
                or (size > 0 and has_base is None)):
            raise ValueError("Invalid replay ring state")
        expected = {"obs": self.obs_dim, "next_obs": self.obs_dim, "action": self.action_dim,
                    "reward": 1, "discount": 1, "terminated": 1, "truncated": 1}
        if has_base:
            expected.update(base_action=HORIZON * POSE_DIM, next_base_action=HORIZON * POSE_DIM)
        if set(state["arrays"]) != set(expected):
            raise ValueError("Replay checkpoint fields differ")
        restored = {}
        for name, width in expected.items():
            dtype = np.bool_ if name in ("terminated", "truncated") else np.float32
            raw = np.asarray(state["arrays"][name])
            if name in ("terminated", "truncated") and raw.dtype != np.bool_:
                raise ValueError("Replay termination flags must be booleans")
            values = _array(raw, (size, width), name, dtype)
            restored[name] = np.empty((self.capacity, width), dtype)
            restored[name][:size] = values
        if (np.any((restored["discount"][:size] < 0) | (restored["discount"][:size] > 1))
                or np.any(np.abs(restored["action"][:size]) > 1.000001)
                or np.any(restored["terminated"][:size] & restored["truncated"][:size])):
            raise ValueError("Invalid replay discount, normalized action or termination flags")
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.arrays, self.rng = restored, rng
        self.size, self.position, self._has_base = size, position, has_base


class _Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, cfg):
        super().__init__()
        self.residual = cfg.method == "action_residual"
        self.log_std_min, self.log_std_max = cfg.log_std_min, cfg.log_std_max
        layers = []
        for width in (obs_dim, cfg.hidden_dim, cfg.hidden_dim):
            layer = nn.Linear(width, cfg.hidden_dim)
            if self.residual:
                nn.init.orthogonal_(layer.weight, math.sqrt(2))
                nn.init.zeros_(layer.bias)
                layers.extend((layer, nn.GELU()))
            else:
                layers.extend((layer, nn.LayerNorm(cfg.hidden_dim), nn.Tanh()))
        self.net = nn.Sequential(*layers)
        self.mean = nn.Linear(cfg.hidden_dim, action_dim)
        self.log_std = nn.Linear(cfg.hidden_dim, action_dim)
        if self.residual:
            for layer in (self.mean, self.log_std):
                nn.init.orthogonal_(layer.weight, .01)
                nn.init.zeros_(layer.bias)

    def forward(self, obs):
        encoded = self.net(obs)
        mean, raw = self.mean(encoded), self.log_std(encoded)
        if self.residual:
            log_std = self.log_std_min + .5 * (self.log_std_max - self.log_std_min) * (raw.tanh() + 1)
        else:
            log_std = raw.clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self(obs)
        normal = torch.distributions.Normal(mean, log_std.exp())
        raw = normal.rsample()
        action = raw.tanh()
        # Evaluate the tanh Jacobian at the pre-squash sample, avoiding atanh(1).
        correction = 2 * (math.log(2) - raw - F.softplus(-2 * raw))
        log_prob = (normal.log_prob(raw) - correction).sum(-1, keepdim=True)
        return action, log_prob, mean.tanh()

    def deterministic(self, obs):
        return self(obs)[0].tanh()


class _EnsembleLinear(nn.Module):
    """ZPRL's independently initialized batched linear layers."""
    def __init__(self, count, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(count, in_dim, out_dim))
        self.bias = nn.Parameter(torch.empty(count, 1, out_dim))
        for i in range(count):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            # Preserve the source's (in,out) matrix initialization convention.
            bound = 1 / math.sqrt(out_dim)
            nn.init.uniform_(self.bias[i], -bound, bound)

    def forward(self, value):
        return torch.bmm(value, self.weight) + self.bias


class _EnsembleLayerNorm(nn.Module):
    def __init__(self, count, width):
        super().__init__()
        self.width = width
        self.weight = nn.Parameter(torch.ones(count, 1, width))
        self.bias = nn.Parameter(torch.zeros(count, 1, width))

    def forward(self, value):
        return F.layer_norm(value, (self.width,)) * self.weight + self.bias


class _Critics(nn.Module):
    def __init__(self, cfg, action_dim):
        super().__init__()
        self.count, self.bounded = cfg.num_qs, cfg.method == "action_residual"
        layers = []
        for width in (cfg.obs_dim + action_dim, cfg.hidden_dim, cfg.hidden_dim):
            layers.append(_EnsembleLinear(cfg.num_qs, width, cfg.hidden_dim))
            if self.bounded:
                layers.append(nn.GELU())
            else:
                layers.extend((_EnsembleLayerNorm(cfg.num_qs, cfg.hidden_dim), nn.Tanh()))
        layers.append(_EnsembleLinear(cfg.num_qs, cfg.hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs, action):
        value = torch.cat((obs, action), dim=-1).unsqueeze(0).expand(self.count, -1, -1)
        logits = self.net(value)
        return .5 * (logits.tanh() + 1) if self.bounded else logits


class SACLearner:
    """CPU/CUDA SAC core, without simulator, HTTP or frozen-policy dependencies.

    ``act`` returns normalized RL actions, never robot pose8. ``update`` accepts
    the exact dictionary returned by ReplayBuffer.sample. The caller supplies
    finite true final observations and next base actions even at termination;
    their Bellman contribution is masked according to the objective.
    """

    def __init__(self, config: LearnerConfig, device="cpu"):
        self.config = copy.deepcopy(config)
        # Revalidate a config if a caller modified a field after construction.
        self.config.__post_init__()
        self.device = torch.device(device)
        self.action_dim = self.config.action_dim
        self.updates = 0
        self.checkpoint_extra: dict[str, Any] = {}
        # torch.manual_seed also seeds every visible CUDA device. Initialize
        # only the CPU generator used for network construction and the single
        # GPU owned by this learner (if any).
        torch.random.default_generator.manual_seed(self.config.seed)
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.manual_seed(self.config.seed)
        self.rng = np.random.default_rng(self.config.seed)
        cfg = self.config
        self.residual = cfg.method == "action_residual"
        # Residual policy outputs can be XYZ-only, but both networks retain the
        # complete frozen pose9 chunk as input, including raw rotation6D.
        critic_action_dim = HORIZON * POSE_DIM if self.residual else self.action_dim
        actor_obs_dim = cfg.obs_dim + (critic_action_dim if self.residual else 0)
        self.actor = _Actor(actor_obs_dim, self.action_dim, cfg).to(self.device)
        self.qs = _Critics(cfg, critic_action_dim).to(self.device)
        self.q_targets = copy.deepcopy(self.qs).requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(cfg.init_alpha), device=self.device))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.policy_lr)
        self.q_optimizer = torch.optim.Adam(self.qs.parameters(), lr=cfg.q_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.q_lr)
        self.scale = torch.tensor(cfg.residual_scale, dtype=torch.float32, device=self.device).repeat(HORIZON)
        self.target_entropy = (cfg.target_entropy if cfg.target_entropy is not None
                               else (-self.action_dim / 2 if self.residual else 0.))

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.detach().exp()) if self.config.auto_alpha else self.config.init_alpha

    def _tensor(self, value, shape, name):
        tensor = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        if tuple(tensor.shape) != tuple(shape) or not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must be finite with shape {shape}, got {tuple(tensor.shape)}")
        return tensor

    def _base_tensor(self, value, batch_size, name):
        if value is None:
            raise ValueError(f"{name} is required for action_residual")
        tensor = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        if tuple(tensor.shape) == (batch_size, HORIZON, POSE_DIM):
            tensor = tensor.flatten(1)
        return self._tensor(tensor, (batch_size, HORIZON * POSE_DIM), name)

    def _actor_input(self, obs, base):
        return torch.cat((obs, base), dim=-1) if self.residual else obs

    def _critic_action(self, action, base):
        if not self.residual:
            return action
        if self.config.residual_mode == "xyz":
            base_chunk = base.reshape(-1, HORIZON, POSE_DIM)
            xyz = base_chunk[..., :3] + (action * self.scale).reshape(-1, HORIZON, 3)
            return torch.cat((xyz, base_chunk[..., 3:]), dim=-1).flatten(1)
        return base + action * self.scale

    @torch.no_grad()
    def act(self, obs, base_actions=None, *, deterministic=False, warmup=False,
            exploration_probability=1.) -> np.ndarray:
        shape = tuple(np.shape(obs))
        if len(shape) != 2 or shape[0] < 1:
            raise ValueError("obs must have shape (B,obs_dim) with B>0")
        batch_size = shape[0]
        obs = self._tensor(obs, (batch_size, self.config.obs_dim), "obs")
        base = self._base_tensor(base_actions, batch_size, "base_actions") if self.residual else None
        if not math.isfinite(exploration_probability) or not 0 <= exploration_probability <= 1:
            raise ValueError("exploration_probability must be in [0,1]")
        if warmup:
            if self.residual:
                actions = torch.zeros((batch_size, self.action_dim), device=self.device)
            else:
                scale = self.config.noise_scale
                actions = torch.randn((batch_size, self.action_dim), device=self.device).clamp(-scale, scale) / scale
        else:
            actor_input = self._actor_input(obs, base)
            actions = (self.actor.deterministic(actor_input) if deterministic
                       else self.actor.sample(actor_input)[0])
            if self.residual and exploration_probability < 1:
                # pi-dec masks whole chunks, not individual action coordinates.
                active = self.rng.random(batch_size) < exploration_probability
                actions[torch.as_tensor(~active, device=self.device)] = 0
        if not torch.isfinite(actions).all():
            raise FloatingPointError("Actor produced a nonfinite normalized action")
        return actions.cpu().numpy().astype(np.float32, copy=True)

    def _prepare_batch(self, batch):
        obs_shape = tuple(np.shape(batch["obs"]))
        if len(obs_shape) != 2 or obs_shape[0] < 1:
            raise ValueError("Batch obs must have shape (B,obs_dim)")
        count = obs_shape[0]
        result = {name: self._tensor(batch[name], (count, width), name)
                  for name, width in (("obs", self.config.obs_dim), ("next_obs", self.config.obs_dim),
                                      ("action", self.action_dim))}
        for name in ("reward", "discount", "terminated", "truncated"):
            values = torch.as_tensor(batch[name], device=self.device)
            if tuple(values.shape) == (count,):
                values = values[:, None]
            result[name] = self._tensor(values, (count, 1), name)
        if (torch.any((result["discount"] < 0) | (result["discount"] > 1))
                or torch.any(result["action"].abs() > 1.000001)):
            raise ValueError("Invalid batch discount or normalized action")
        for name in ("terminated", "truncated"):
            if torch.any((result[name] != 0) & (result[name] != 1)):
                raise ValueError("Termination flags must be binary")
        if torch.any(result["terminated"] * result["truncated"]):
            raise ValueError("A transition cannot be both terminated and truncated")
        if self.residual:
            for name in ("base_action", "next_base_action"):
                result[name] = self._base_tensor(batch.get(name), count, name)
        else:
            result["base_action"] = result["next_base_action"] = None
        return result

    @torch.no_grad()
    def _bellman_target(self, batch):
        """Return (B,1) targets using actual duration and explicit truncation policy."""
        next_action, log_prob, _ = self.actor.sample(
            self._actor_input(batch["next_obs"], batch["next_base_action"]))
        values = self.q_targets(batch["next_obs"], self._critic_action(next_action, batch["next_base_action"]))
        subset = torch.randperm(self.config.num_qs, device=self.device)[:self.config.num_subset]
        value = values[subset].min(0).values
        if not self.residual and self.config.q_entropy:
            value = value - self.alpha * log_prob
        bootstrap = 1 - batch["terminated"]
        if not self.config.bootstrap_truncated:
            bootstrap = bootstrap * (1 - batch["truncated"])
        return batch["reward"] + bootstrap * batch["discount"] * value

    def _actor_loss(self, batch):
        actor_input = self._actor_input(batch["obs"], batch["base_action"])
        action, log_prob, mean = self.actor.sample(actor_input)
        values = self.qs(batch["obs"], self._critic_action(action, batch["base_action"]))
        value = values.mean(0) if self.residual else values.min(0).values
        rl_loss = (self.alpha * log_prob - value).mean()
        spatial = temporal = rl_loss.new_zeros(())
        if self.residual and self.config.lambda_s is not None:
            noisy_input = (actor_input + self.config.sigma * torch.randn_like(actor_input)).detach()
            noisy_mean = self.actor.deterministic(noisy_input)
            combined = self._critic_action(mean, batch["base_action"])
            noisy_combined = self._critic_action(noisy_mean, batch["base_action"])
            spatial = .5 * (noisy_combined - combined).square().sum(-1).mean()
            chunk = combined.reshape(-1, HORIZON, POSE_DIM)
            temporal = .5 * (chunk[:, 1:] - chunk[:, :-1]).square().sum(-1).mean()
        loss = rl_loss + (self.config.lambda_s or 0.) * spatial + (self.config.lambda_t or 0.) * temporal
        return loss, {"actor_rl_loss": float(rl_loss.detach()),
                      "spatial_smoothness_loss": float(spatial.detach()),
                      "temporal_smoothness_loss": float(temporal.detach()),
                      "actor_entropy": float(-log_prob.detach().mean()),
                      "normalized_action_norm": float(action.detach().norm(dim=-1).mean())}

    def update(self, batch: dict) -> dict[str, float]:
        values = self._prepare_batch(batch)
        target = self._bellman_target(values)
        current = self.qs(values["obs"], self._critic_action(values["action"], values["base_action"]))
        loss = (current - target.unsqueeze(0)).square().mean(dim=(1, 2)).sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite critic loss")
        self.q_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        critic_norm = nn.utils.clip_grad_norm_(self.qs.parameters(), self.config.max_grad_norm,
                                              error_if_nonfinite=True)
        self.q_optimizer.step()
        self.updates += 1
        target_updated = self.updates % self.config.target_freq == 0
        if target_updated:
            with torch.no_grad():
                for target_param, param in zip(self.q_targets.parameters(), self.qs.parameters(), strict=True):
                    target_param.lerp_(param, self.config.tau)
        metrics = {"updates": float(self.updates), "critic_loss": float(loss.detach()) / self.config.num_qs,
                   "q_target": float(target.mean()), "q_predicted": float(current.detach().mean()),
                   "critic_grad_norm": float(critic_norm), "reward": float(values["reward"].mean()),
                   "discount": float(values["discount"].mean()), "target_updated": float(target_updated),
                   "actor_updated": 0.}
        if self.updates % self.config.policy_freq == 0:
            # Q remains differentiable with respect to action, but its weights
            # receive no actor gradients and are never stepped by actor_opt.
            self.qs.requires_grad_(False)
            try:
                actor_loss, actor_metrics = self._actor_loss(values)
                if not torch.isfinite(actor_loss):
                    raise FloatingPointError("Nonfinite actor loss")
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm,
                                                     error_if_nonfinite=True)
                self.actor_optimizer.step()
            finally:
                self.qs.requires_grad_(True)
            metrics.update(actor_metrics, actor_loss=float(actor_loss.detach()),
                           actor_grad_norm=float(actor_norm), actor_updated=1.)
            if self.config.auto_alpha:
                with torch.no_grad():
                    _, log_prob, _ = self.actor.sample(self._actor_input(values["obs"], values["base_action"]))
                alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy)).mean()
                self.alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_optimizer.step()
                metrics["alpha_loss"] = float(alpha_loss.detach())
        metrics["alpha"] = self.alpha
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("Nonfinite learner metrics")
        return metrics

    def save(self, path, extra: dict | None = None):
        """Atomically save a trusted local checkpoint, including all update/RNG state.

        Replay is deliberately caller-owned: put replay.state_dict() in extra
        when exact replay continuation is desired. Simulator continuation still
        needs the explicit fresh-episode boundary described by the protocol.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": CHECKPOINT_VERSION, "source_commit": SOURCE_COMMIT,
                   "config": asdict(self.config), "updates": self.updates,
                   "actor": self.actor.state_dict(), "qs": self.qs.state_dict(),
                   "q_targets": self.q_targets.state_dict(), "log_alpha": self.log_alpha.detach(),
                   "actor_optimizer": self.actor_optimizer.state_dict(),
                   "q_optimizer": self.q_optimizer.state_dict(),
                   "alpha_optimizer": self.alpha_optimizer.state_dict(),
                   "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                           "python": random.getstate(), "learner": self.rng.bit_generator.state,
                           # This process owns one learner device. Never query
                           # other GPUs, which may belong to the simulator.
                           "cuda": {"device_index": (self.device.index if self.device.index is not None
                                                     else torch.cuda.current_device()),
                                    "state": torch.cuda.get_rng_state(self.device)}
                           if self.device.type == "cuda" else None},
                   "extra": {} if extra is None else extra}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
                temporary = Path(stream.name)
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path, device="cpu") -> "SACLearner":
        """Load only a trusted checkpoint: optimizer/RNG/extra use pickle data."""
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        if payload.get("version") != CHECKPOINT_VERSION or payload.get("source_commit") != SOURCE_COMMIT:
            raise ValueError("Unsupported learner checkpoint identity")
        learner = cls(LearnerConfig(**payload["config"]), device=device)
        learner.actor.load_state_dict(payload["actor"], strict=True)
        learner.qs.load_state_dict(payload["qs"], strict=True)
        learner.q_targets.load_state_dict(payload["q_targets"], strict=True)
        learner.q_targets.requires_grad_(False)
        with torch.no_grad():
            learner.log_alpha.copy_(payload["log_alpha"])
        learner.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        learner.q_optimizer.load_state_dict(payload["q_optimizer"])
        learner.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        if type(payload["updates"]) is not int or payload["updates"] < 0:
            raise ValueError("Invalid checkpoint update count")
        learner.updates = payload["updates"]
        learner.checkpoint_extra = payload["extra"]
        rng = payload["rng"]
        torch.set_rng_state(rng["torch"].cpu())
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        learner.rng.bit_generator.state = rng["learner"]
        if learner.device.type == "cuda" and rng["cuda"] is not None:
            # A checkpoint can move between physical GPU ordinals/counts;
            # restore the owned generator on the device selected for this run.
            torch.cuda.set_rng_state(rng["cuda"]["state"].cpu(), device=learner.device)
        return learner
