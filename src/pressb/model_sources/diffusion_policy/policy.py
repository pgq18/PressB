"""Visual Diffusion Policy with the actual DPPO conditional temporal UNet.

The ResNet/spatial-softmax encoder is a PressB integration. DDPM schedules,
denoising equations and the UNet are imported from the pinned MIT DPPO source.
Cached conditioning separates the frozen observation encoder from the action
expert that is fine-tuned by DPPO.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from threading import RLock
import torch
from torch import nn
import torch.nn.functional as F

from .diffusion import DiffusionModel
from .unet import Unet1D


_PRECISION_LOCK = RLock()


@contextmanager
def _ieee_float32_transition(device):
    """Keep the rollout and PPO likelihood independent of TF32 kernel choice.

    cuDNN can choose different TF32 convolutions for the small rollout batch
    and a large, mixed-timestep PPO batch. Their mean errors become material
    likelihood errors at small transition sigma. Scope the flags to this
    expert's forward pass; preserve caller settings and allow autograd.
    """
    with _PRECISION_LOCK, torch.autocast(device_type=device.type, enabled=False):
        if device.type != "cuda":
            yield
            return
        matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.backends.cudnn.flags(
                enabled=torch.backends.cudnn.enabled,
                benchmark=torch.backends.cudnn.benchmark,
                benchmark_limit=torch.backends.cudnn.benchmark_limit,
                deterministic=torch.backends.cudnn.deterministic,
                allow_tf32=False,
            ):
                yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = matmul_tf32


class SpatialSoftmax(nn.Module):
    def __init__(self, channels, keypoints):
        super().__init__()
        self.project = nn.Conv2d(channels, keypoints, 1)

    def forward(self, x):
        x = self.project(x)
        height, width = x.shape[-2:]
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, height, device=x.device),
                                torch.linspace(-1, 1, width, device=x.device), indexing="ij")
        weights = x.flatten(2).softmax(-1)
        return torch.cat(((weights * xx.flatten()).sum(-1),
                          (weights * yy.flatten()).sum(-1)), dim=-1)


class VisualEncoder(nn.Module):
    def __init__(self, cameras, state_dim, tasks, keypoints=32, task_dim=32):
        super().__init__()
        from torchvision.models import resnet18

        # GroupNorm supports small per-GPU batches and does not update hidden
        # BatchNorm statistics while serving cached observations.
        network = resnet18(weights=None, norm_layer=lambda n: nn.GroupNorm(min(n, 32), n))
        self.backbone = nn.Sequential(*list(network.children())[:-2])
        self.spatial = SpatialSoftmax(512, keypoints)
        self.task_embedding = nn.Embedding(tasks, task_dim)
        self.feature_dim = len(cameras) * keypoints * 2 + state_dim + task_dim
        self.register_buffer("image_mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("image_std", torch.tensor([.229, .224, .225])[None, :, None, None])

    def forward(self, images, state, task_id):
        b, n = images.shape[:2]
        image = images.flatten(0, 1).float() / 255
        image = (image - self.image_mean) / self.image_std
        visual = self.spatial(self.backbone(image)).reshape(b, -1)
        return torch.cat((visual, state, self.task_embedding(task_id)), dim=-1)


class VisualDiffusionPolicy(nn.Module):
    def __init__(self, config, normalization, device="cpu"):
        super().__init__()
        self.config = deepcopy(config)
        self.horizon = int(config["horizon"])
        self.action_dim = len(config["action_names"])
        self.encoder = VisualEncoder(config["cameras"], len(config["state_names"]),
                                    len(config["tasks"]), config.get("keypoints", 32),
                                    config.get("task_embedding_dim", 32))
        self.feature_dim = self.encoder.feature_dim
        mults = tuple(config.get("unet_dim_mults", [1, 2, 4]))
        if self.horizon % (2 ** (len(mults) - 1)):
            raise ValueError("horizon must be divisible by the temporal UNet downsampling factor")
        network = Unet1D(action_dim=self.action_dim, cond_dim=self.feature_dim,
                         diffusion_step_embed_dim=config.get("time_embedding_dim", 64),
                         dim=config.get("unet_dim", 64), dim_mults=mults,
                         smaller_encoder=True, n_groups=8, cond_predict_scale=True)
        self.diffusion = DiffusionModel(network, self.horizon, self.feature_dim, self.action_dim,
                                       device=device, denoising_steps=config.get("diffusion_steps", 100),
                                       denoised_clip_value=1.0, predict_epsilon=True)
        for key in ("state_center", "state_scale", "action_center", "action_scale"):
            self.register_buffer(key, torch.tensor(normalization[key], dtype=torch.float32))
        self.to(device)

    def encode(self, images, states_si, task_ids):
        return self.encoder(images, (states_si - self.state_center) / self.state_scale, task_ids)

    def loss(self, images, states_si, task_ids, actions_si, *, timesteps=None, noise=None):
        context = self.encode(images, states_si, task_ids)
        normalized = (actions_si - self.action_center) / self.action_scale
        if timesteps is None and noise is None:
            return self.diffusion.loss(normalized, {"state": context})
        if timesteps is None or noise is None:
            raise ValueError("Provide both timesteps and noise for deterministic validation")
        noisy = self.diffusion.q_sample(normalized, timesteps, noise=noise)
        prediction = self.diffusion.network(noisy, timesteps, {"state": context})
        return F.mse_loss(prediction, noise)

    def actions_from_latent(self, latent):
        return latent * self.action_scale + self.action_center

    def action_expert(self, config=None):
        return DiffusionActionExpert(self, config or {})


class DiffusionActionExpert(nn.Module):
    """Head-only, differentiable Gaussian reverse chain for DPPO.

    DDIM uses the upstream leading time grid and cosine schedule. A nonzero
    fixed standard deviation floor also applies at the final denoising step;
    the rollout and likelihood code must use this same distribution. No latent
    clipping is performed after sampling (the reconstructed clean prediction is
    clipped inside the native DDPM/DDIM mean, as in upstream).
    """
    def __init__(self, policy, config):
        super().__init__()
        allowed = {"kernel", "denoising_steps", "eta", "min_std"}
        if not isinstance(config, dict) or set(config) - allowed:
            raise ValueError(f"Diffusion action expert supports {sorted(allowed)}")
        self.network = policy.diffusion.network
        if any(parameter.dtype != torch.float32 for parameter in self.network.parameters()):
            raise ValueError("Diffusion action expert requires float32 network parameters")
        self.horizon, self.latent_dim = policy.horizon, policy.action_dim
        self.kernel = config.get("kernel", "ddim")
        self.denoising_steps = config.get("denoising_steps", policy.config.get("inference_steps", 10))
        if type(self.denoising_steps) is not int:
            raise ValueError("denoising_steps must be an integer")
        total = policy.diffusion.denoising_steps
        if self.kernel not in ("ddpm", "ddim") or not 1 <= self.denoising_steps <= total:
            raise ValueError("Invalid reverse diffusion kernel or denoising step count")
        if self.kernel == "ddpm" and self.denoising_steps != total:
            raise ValueError("DDPM requires all trained diffusion steps; use DDIM to subsample")
        self.eta = float(config.get("eta", 1.0))
        self.min_std = float(config.get("min_std", 0.01))
        if not 0 <= self.eta <= 1 or not 0 <= self.min_std <= 1:
            raise ValueError("eta and min_std must be in [0,1]")
        grid = (torch.arange(self.denoising_steps) * (total // self.denoising_steps)).flip(0)
        if self.kernel == "ddpm":
            grid = torch.arange(total).flip(0)
        cumulative = policy.diffusion.alphas_cumprod.detach().cpu()
        self.register_buffer("timesteps", grid)
        self.register_buffer("alphas", cumulative[grid])
        self.register_buffer("previous_alphas", torch.cat((cumulative[grid[1:]], torch.ones(1))))
        for key in ("action_center", "action_scale"):
            self.register_buffer(key, getattr(policy, key).detach().clone())
        if self.kernel == "ddpm":
            for key in ("ddpm_mu_coef1", "ddpm_mu_coef2", "ddpm_logvar_clipped"):
                self.register_buffer(key, getattr(policy.diffusion, key).detach().clone())
        self.to(next(self.network.parameters()).device)

    def transition(self, x, step, conditioning):
        with _ieee_float32_transition(x.device):
            return self._transition_float32(x.float(), step, conditioning)

    def _transition_float32(self, x, step, conditioning):
        step = torch.as_tensor(step, device=x.device, dtype=torch.long).expand(len(x))
        t = self.timesteps[step]
        epsilon = self.network(x, t, {"state": conditioning["features"].float()}).float()
        alpha = self.alphas[step, None, None]
        previous = self.previous_alphas[step, None, None]
        reconstructed = ((x.float() - (1 - alpha).sqrt() * epsilon) / alpha.sqrt()).clamp(-1, 1)
        if self.kernel == "ddpm":
            mean = self.ddpm_mu_coef1[t, None, None] * reconstructed + self.ddpm_mu_coef2[t, None, None] * x
            std = (self.ddpm_logvar_clipped[t, None, None] * .5).exp()
        else:
            epsilon = (x.float() - alpha.sqrt() * reconstructed) / (1 - alpha).sqrt()
            variance = ((1 - previous) / (1 - alpha) * (1 - alpha / previous)).clamp_min(0)
            std = self.eta * variance.sqrt()
            mean = previous.sqrt() * reconstructed + (1 - previous - std.square()).clamp_min(0).sqrt() * epsilon
        return mean.float(), std.clamp_min(self.min_std).expand_as(x).float()

    @property
    def device(self):
        return next(self.network.parameters()).device

    def actions_from_latent(self, latent):
        return latent * self.action_scale + self.action_center

    def spec(self):
        return {"kind": "diffusion_policy", "kernel": self.kernel,
                "denoising_steps": self.denoising_steps, "eta": self.eta,
                "min_std": self.min_std, "horizon": self.horizon,
                "latent_dim": self.latent_dim, "prediction": "epsilon",
                "beta_schedule": "upstream_cosine", "clip_reconstructed_clean": 1.0,
                "post_sample_clip": False, "transition_forward_precision": "ieee_float32",
                "transition_autocast": False, "transition_tf32": False}

    def describe(self):
        return self.spec()

    def pack_contexts(self, contexts):
        return {"features": torch.stack([context.features.detach().clone() for context in contexts])}
