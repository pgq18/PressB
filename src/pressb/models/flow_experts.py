"""Trainable action experts for the stochastic-flow extension of DPPO.

The existing VLA-JEPA and Wall-OSS policies are deterministic Euler flow
samplers, not DDPM policies. These wrappers define a *new* Gaussian transition
kernel with the existing velocity as its mean drift and a configured fixed
standard deviation. Sampling and PPO scoring must both use this same kernel.
It is not claimed to preserve the pretrained flow's marginal distributions.

Only actual action-head parameters are registered on these modules. Encoders,
normalizers and Wall's shared attention/text expert remain frozen; detached
observation embeddings stay valid after an action-expert update. Wall always
recomputes the transformer and never reuses parameter-dependent KV caches.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn


def _settings(config, checkpoint_steps):
    config = {} if config is None else dict(config)
    unknown = set(config) - {"transition_std", "denoising_steps"}
    if unknown:
        raise ValueError(f"Unknown stochastic-flow expert settings: {sorted(unknown)}")
    steps = config.get("denoising_steps", checkpoint_steps)
    if type(steps) is not int or steps < 1:
        raise ValueError("denoising_steps must be a positive integer")
    std = config.get("transition_std", 0.05)
    if isinstance(std, bool) or not isinstance(std, (float, int)) or not math.isfinite(std) or std <= 0:
        raise ValueError("transition_std must be finite and strictly positive, including the final step")
    return steps, float(std)


def _cpu_copy(value):
    # Encoder caches may be inference tensors. Clone outside inference mode so
    # autograd can save the resulting conditioning in the trainer's backward.
    with torch.inference_mode(False):
        return value.detach().to(device="cpu").clone()


class _FlowExpert(nn.Module):
    sampler_kind = "stochastic_flow_euler_dppo_extension"

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def train(self, mode=True):
        # Training gradients do not require stochastic dropout. Keep rollout
        # and PPO re-scoring numerically identical for identical inputs.
        super().train(False)
        self._source_model.eval()
        return self

    def _check_transition(self, x, step, conditioning):
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.horizon, self.latent_dim):
            raise ValueError("Latent must be [batch, horizon, latent_dim]")
        if step.shape != (x.shape[0],) or step.dtype not in (torch.int32, torch.int64):
            raise ValueError("step must contain one integer denoising index per batch row")
        if torch.any(step < 0) or torch.any(step >= self.denoising_steps):
            raise ValueError("Denoising step is outside the configured chain")
        if not isinstance(conditioning, Mapping):
            raise ValueError("conditioning must be a tensor mapping")

    def describe(self):
        return dict(
            sampler_kind=self.sampler_kind,
            denoising_steps=self.denoising_steps, horizon=self.horizon,
            latent_dim=self.latent_dim, transition_std=self.transition_std,
            chain_dtype="float32", velocity_dtype=str(self.dtype),
            observation_encoder_frozen=True, transformer_kv_cache=False,
            trainable_parameter_count=sum(p.numel() for p in self.parameters()),
            trainable_source_parameters=list(self.trainable_source_parameters),
            deterministic_evaluation="Euler mean with FP32 accumulation and no transition noise",
        )

    def assert_frozen_encoder(self):
        trainable_ids = {id(parameter) for parameter in self.parameters()}
        violations = [name for name, parameter in self._source_model.named_parameters()
                      if parameter.requires_grad and id(parameter) not in trainable_ids]
        if violations:
            raise RuntimeError(f"Parameters outside the action expert are trainable: {violations}")

    def collate_conditions(self, conditions):
        if not conditions or any(set(row) != set(conditions[0]) for row in conditions):
            raise ValueError("Expected nonempty conditioning rows with identical keys")
        return {key: torch.cat([row[key] for row in conditions], dim=0) for key in conditions[0]}


class VLAJEPAFlowExpert(_FlowExpert):
    """GR00T flow action head, with embodied tokens and state as conditioning."""

    def __init__(self, model, *, action_scale=None, config=None):
        super().__init__()
        model.eval().requires_grad_(False)
        object.__setattr__(self, "_source_model", model)
        self.head = model.action_model
        self.head.requires_grad_(True)
        self.horizon, self.latent_dim = self.head.action_horizon, self.head.action_dim
        self.denoising_steps, self.transition_std = _settings(config, self.head.num_inference_timesteps)
        scale = torch.ones(self.latent_dim) if action_scale is None else torch.as_tensor(action_scale).float()
        if scale.shape != (self.latent_dim,) or not torch.isfinite(scale).all() or (scale == 0).any():
            raise ValueError("action_scale must contain one finite, nonzero scale per latent channel")
        self.register_buffer("action_scale", scale, persistent=False)
        self.trainable_source_parameters = tuple("action_model." + name for name, _ in self.head.named_parameters())
        self.train(False)
        self.assert_frozen_encoder()

    def pack_contexts(self, contexts):
        if not contexts:
            raise ValueError("At least one encoder context is required")
        contexts = [getattr(context, "encoded", context) for context in contexts]
        return {key: _cpu_copy(torch.cat([getattr(context, key) for context in contexts], dim=0))
                for key in ("tokens", "state")}

    def transition(self, x, step, conditioning):
        self._check_transition(x, step, conditioning)
        head = self.head
        x = x.to(device=self.device, dtype=torch.float32)
        tokens = conditioning["tokens"].detach().to(device=self.device, dtype=self.dtype)
        state = conditioning["state"].detach().to(device=self.device, dtype=self.dtype)
        step = step.to(device=self.device)
        # Python int(k / K * buckets) in the original sampler, including
        # non-divisible bucket/step counts (float64 avoids fp32 floor drift).
        times = (step.double() / self.denoising_steps * head.num_timestep_buckets).long()
        with torch.autocast(device_type=self.device.type, enabled=False):
            features = head.action_encoder(x.to(dtype=self.dtype), times)
            if head.config.add_pos_embed:
                positions = torch.arange(features.shape[1], dtype=torch.long, device=self.device)
                features = features + head.position_embedding(positions).unsqueeze(0)
            states = head._encode_state(state, x.shape[0])
            future = head.future_tokens.weight.unsqueeze(0).expand(x.shape[0], -1, -1)
            inputs = torch.cat((states, future, features), dim=1) if states is not None else torch.cat((future, features), dim=1)
            hidden = head.model(hidden_states=inputs, encoder_hidden_states=tokens, timestep=times)
            velocity = head.action_decoder(hidden)[:, -self.horizon:]
        mean = x + velocity.float() / self.denoising_steps
        return mean, torch.full_like(mean, self.transition_std)

    def actions_from_latent(self, x):
        return x.float() * self.action_scale.to(device=x.device)


class WallOSSFlowExpert(_FlowExpert):
    """Wall action-route MLPs and action projections, with full 20-D latents.

    Proprioception projection is deliberately frozen: it contributes to the
    cached embeddings. Shared attention/norms and the text MoE expert are also
    frozen, though gradients pass through their operations to the action MLPs.
    """

    def __init__(self, backend, *, config=None):
        super().__init__()
        config = {} if config is None else dict(config)
        self.transition_microbatch_size = config.pop("transition_microbatch_size", 1)
        self.transition_context_length = config.pop("transition_context_length", None)
        if type(self.transition_microbatch_size) is not int or self.transition_microbatch_size not in (1, 8):
            raise ValueError("Wall DPPO transition_microbatch_size must be 1 or 8")
        if self.transition_microbatch_size == 8:
            if type(self.transition_context_length) is not int or self.transition_context_length < 1:
                raise ValueError("Wall batch 8 requires an explicit positive transition_context_length")
        elif self.transition_context_length is not None:
            raise ValueError("transition_context_length is only used with fixed batch 8")
        model = backend.policy.model
        model.eval().requires_grad_(False)
        object.__setattr__(self, "_source_model", model)
        self.horizon, self.latent_dim = backend.horizon, backend.noise_dim
        self.action_dim, self.dataset_name = backend.dimension, backend.dataset_name
        self.denoising_steps, self.transition_std = _settings(config, backend.num_inference_timesteps)
        if getattr(model.config, "_attn_implementation", "eager") != "eager":
            raise ValueError("Wall DPPO requires eager attention for independent padded batches")
        self.action_token_id = model.action_token_id_set["action_token_id"]
        modules = {}
        for index, layer in enumerate(model.model.layers):
            moe = getattr(layer, "moe", None)
            if moe is None or len(moe.experts) != 2:
                raise ValueError("Wall DPPO requires the native two-route MoE action expert")
            modules[f"layer_{index}"] = moe.experts[1]
        if not modules:
            raise ValueError("Wall action expert contains no transformer layers")
        for name in ("w1", "w2", "w3", "action_proj_back"):
            modules[name] = getattr(model.action_preprocessor, name)
        self.expert_modules = nn.ModuleDict(modules)
        self.expert_modules.requires_grad_(True)
        selected = {id(parameter) for parameter in self.expert_modules.parameters()}
        self.trainable_source_parameters = tuple(name for name, parameter in model.named_parameters() if id(parameter) in selected)
        scale = torch.as_tensor(backend.action_scale).float()
        self.register_buffer("action_scale", scale, persistent=False)
        self.train(False)
        self.assert_frozen_encoder()

    def describe(self):
        fixed = self.transition_microbatch_size == 8
        return dict(super().describe(), transition_microbatch_size=self.transition_microbatch_size,
                    transition_context_length=self.transition_context_length,
                    context_padding="fixed_left_padding_reject_overflow" if fixed else "trim_each_row_before_every_transition",
                    transition_batching="fixed_batch8_duplicate_tail_discard_outputs" if fixed else "serial_action_transformer; observation_encoder_can_batch")

    def pack_contexts(self, contexts):
        if not contexts:
            raise ValueError("At least one encoder context is required")
        from wall_x.model.flow_context import batch_diffusion_context
        rows = [context.conditioning for context in contexts]
        if any(row["inputs"]["dataset_names"] != [self.dataset_name] for row in rows):
            raise ValueError("Wall context normalizer does not match the expert")
        packed = batch_diffusion_context(rows)["inputs"]
        # Every payload tensor has batch as its leading dimension, including
        # RoPE positions, so trainer shuffling/slicing cannot exchange rows.
        result = {key: _cpu_copy(packed[key]) for key in
                  ("input_ids", "inputs_embeds", "attention_mask", "moe_token_types", "dof_mask")}
        result["position_ids"] = _cpu_copy(packed["position_ids"].transpose(0, 1))
        return result

    def transition(self, x, step, conditioning):
        self._check_transition(x, step, conditioning)
        # BF16 GEMM kernels differ between B=1 and B=8 even at identical
        # weights. A small velocity difference can become a large density
        # ratio. Keep both batch size and token shape identical between
        # sampling and shuffled PPO re-scoring. Fixed B8 also pads unused tail
        # rows; these outputs never enter the loss or the returned actions.
        means, stds = [], []
        size = self.transition_microbatch_size
        for first in range(0, x.shape[0], size):
            count = min(size, x.shape[0] - first)
            rows = [self._canonical_condition_row(conditioning, index)
                    for index in range(first, first + count)]
            rows.extend([rows[-1]] * (size - count))
            packed = {key: torch.cat([row[key] for row in rows], dim=0) for key in rows[0]}
            batch_x, batch_step = x[first:first + count], step[first:first + count]
            if count < size:
                batch_x = torch.cat((batch_x, batch_x[-1:].expand(size - count, -1, -1)), dim=0)
                batch_step = torch.cat((batch_step, batch_step[-1:].expand(size - count)), dim=0)
            mean, std = self._transition_row(batch_x, batch_step, packed)
            means.append(mean[:count])
            stds.append(std[:count])
        return torch.cat(means, dim=0), torch.cat(stds, dim=0)

    def _canonical_condition_row(self, conditioning, index):
        row = {key: value[index:index + 1] for key, value in conditioning.items()}
        mask = row["attention_mask"]
        if mask.ndim != 2 or mask.shape[0] != 1:
            raise ValueError("Wall conditioning requires batch-first two-dimensional attention_mask")
        count = int(mask.bool().sum())
        start = mask.shape[1] - count
        if count == 0 or mask[:, :start].bool().any() or not mask[:, start:].bool().all():
            raise ValueError("Wall conditioning must have contiguous valid tokens after left padding")
        for key in ("input_ids", "inputs_embeds", "attention_mask", "moe_token_types"):
            row[key] = row[key][:, start:]
        row["position_ids"] = row["position_ids"][:, :, start:]
        if self.transition_context_length is not None:
            pad = self.transition_context_length - count
            if pad < 0:
                raise ValueError(f"Wall context has {count} valid tokens, exceeding fixed transition_context_length={self.transition_context_length}")
            for key, value in row.items():
                if key == "inputs_embeds":
                    row[key] = torch.nn.functional.pad(value, (0, 0, pad, 0))
                elif key != "dof_mask":
                    row[key] = torch.nn.functional.pad(value, (pad, 0), value=1 if key == "position_ids" else 0)
        return row

    def _transition_row(self, x, step, conditioning):
        model = self._source_model
        x = x.to(device=self.device, dtype=torch.float32)
        values = {key: value.detach().to(device=self.device) for key, value in conditioning.items()}
        embeds = values["inputs_embeds"].to(dtype=self.dtype)
        ids = values["input_ids"]
        mask = ids == self.action_token_id
        routes = values["moe_token_types"].long()
        if not torch.all(mask.sum(dim=1) == self.horizon):
            raise ValueError("Each Wall context must have horizon action tokens")
        if (routes[mask] != 1).any() or (routes[~mask] != 0).any():
            raise ValueError("Wall action-only updates require action tokens routed to expert 1 exclusively")
        if (mask & ~values["attention_mask"].bool()).any():
            raise ValueError("Wall action tokens cannot be padded")
        counts = torch.bincount(routes.flatten(), minlength=2).cpu()
        ends = counts.cumsum(0)
        times = torch.linspace(0, 1, self.denoising_steps + 1, device=self.device, dtype=self.dtype)
        indices = step.to(device=self.device)
        t = times[indices]
        dt = (times[indices + 1] - t).float()[:, None, None]
        with torch.autocast(device_type=self.device.type, enabled=False):
            action_embed = model.action_preprocessor.step(
                timestep=t, noisy_action=x.to(dtype=self.dtype), dof_mask=values["dof_mask"].to(dtype=self.dtype))
            inputs = embeds.clone()
            inputs[mask] = action_embed.reshape(-1, embeds.shape[-1])
            output = model.model(
                input_ids=None, attention_mask=values["attention_mask"],
                position_ids=values["position_ids"].transpose(0, 1),
                inputs_embeds=inputs, moe_token_types=routes,
                start_indices=ends - counts, end_indices=ends,
                past_key_values=None, use_cache=False, output_attentions=False,
                output_hidden_states=False, return_dict=True,
            )
            velocity = model.action_preprocessor.action_proj_back(output.last_hidden_state[mask])
            velocity = velocity.reshape(x.shape[0], self.horizon, self.latent_dim)
        mean = x + dt * velocity.float()
        return mean, torch.full_like(mean, self.transition_std)

    def collate_conditions(self, conditions):
        if not conditions or any(set(row) != set(conditions[0]) for row in conditions):
            raise ValueError("Expected nonempty conditioning rows with identical keys")
        length = max(row["input_ids"].shape[1] for row in conditions)
        result = {}
        for key in conditions[0]:
            rows = []
            for condition in conditions:
                value = condition[key]
                pad = length - condition["input_ids"].shape[1]
                if key == "inputs_embeds":
                    value = torch.nn.functional.pad(value, (0, 0, pad, 0))
                elif key != "dof_mask":
                    value = torch.nn.functional.pad(value, (pad, 0), value=1 if key == "position_ids" else 0)
                rows.append(value)
            result[key] = torch.cat(rows, dim=0)
        return result

    def actions_from_latent(self, x):
        native = self._source_model.action_preprocessor.normalizer_action.unnormalize_data(
            x.float(), [self.dataset_name] * x.shape[0])
        return native[:, :, :self.action_dim].float() * self.action_scale.to(device=x.device)


def create_vlajepa_expert(backend, config=None):
    if hasattr(backend, "frozen"):
        return VLAJEPAFlowExpert(backend.frozen.model, config=config)
    from .vlajepa_joint import MODEL_SCALE, MODEL_TO_SI
    scale = MODEL_SCALE[:backend.dimension] * MODEL_TO_SI[:backend.dimension]
    return VLAJEPAFlowExpert(backend.model, action_scale=scale, config=config)


def create_walloss_expert(backend, config=None):
    return WallOSSFlowExpert(backend, config=config)
