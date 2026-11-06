"""Small, dependency-light helpers for externally controlled flow inference."""

import torch


def initial_flow_noise(initial_noise, *, batch_size, horizon, action_dim, reference):
    """Use caller noise verbatim; None retains the original torch RNG behavior.

    The full internal action width is required, including padded channels: the
    action embedding consumes those channels even when the output mask hides them.
    """
    shape = (batch_size, horizon, action_dim)
    if initial_noise is None:
        return torch.randn(size=shape, dtype=reference.dtype, device=reference.device)
    noise = torch.as_tensor(initial_noise, device=reference.device)
    if tuple(noise.shape) != shape:
        raise ValueError(f"initial_noise must have shape {shape}, got {tuple(noise.shape)}")
    if not noise.is_floating_point() or not torch.isfinite(noise).all():
        raise ValueError("initial_noise must contain finite floating point values")
    noise = noise.to(dtype=reference.dtype)
    if not torch.isfinite(noise).all():
        raise ValueError("initial_noise overflows the model dtype")
    return noise.detach().clone()


def pool_conditioning_feature(inputs_embeds, input_ids, action_token_id, attention_mask=None):
    """Mean multimodal prefix embedding, excluding actions and padding.

    This is a noise-independent feature of vision, prompt and proprioception;
    it is not a transformer hidden-state or KV-cache feature.
    """
    valid = input_ids.ne(action_token_id)
    if attention_mask is not None:
        if attention_mask.ndim != 2:
            raise ValueError("Context features require a 2-D attention mask")
        valid = valid & attention_mask.bool()
    if not valid.any(dim=1).all():
        raise ValueError("Observation has no valid conditioning tokens")
    weights = valid.to(torch.float32).unsqueeze(-1)
    return (inputs_embeds.float() * weights).sum(dim=1) / weights.sum(dim=1)


_CONTEXT_SCALARS = ("pred_horizon", "action_dim", "num_inference_timesteps")
_CONTEXT_SEQUENCES = ("input_ids", "inputs_embeds", "attention_mask", "moe_token_types")
_CONTEXT_KEYS = set(_CONTEXT_SCALARS + _CONTEXT_SEQUENCES + ("position_ids", "dof_mask", "dataset_names"))


def split_diffusion_context(context):
    """Split a native batch into independently owned, unpadded cache entries.

    Clone slices so a long-lived single cache entry neither retains the full
    batch allocation nor aliases another entry. RoPE coordinates of valid
    tokens are kept verbatim; padding is not an observation feature.
    """
    inputs = context["inputs"]
    _check_context_keys(inputs)
    embeds = inputs["inputs_embeds"]
    batch_size, length = embeds.shape[:2]
    mask = inputs["attention_mask"]
    if mask is None:
        mask = torch.ones((batch_size, length), dtype=torch.long, device=embeds.device)
    if tuple(mask.shape) != (batch_size, length):
        raise ValueError("Context requires a two-dimensional attention mask")
    names = _dataset_names(inputs["dataset_names"], batch_size)
    rows = []
    for index in range(batch_size):
        valid = mask[index].bool()
        count = int(valid.sum().item())
        start = length - count
        if count == 0 or not valid[start:].all() or valid[:start].any():
            raise ValueError("Context batching requires contiguous left padding")
        row = {key: inputs[key] for key in _CONTEXT_SCALARS}
        for key in _CONTEXT_SEQUENCES:
            value = mask if key == "attention_mask" else inputs[key]
            if value is None:
                raise ValueError(f"Context {key} is required for batching")
            row[key] = value[index:index + 1, start:].detach().clone()
        position_ids = inputs["position_ids"]
        if position_ids is None or tuple(position_ids.shape) != (3, batch_size, length):
            raise ValueError("Context requires 3-axis RoPE positions")
        row["position_ids"] = position_ids[:, index:index + 1, start:].detach().clone()
        row["dof_mask"] = inputs["dof_mask"][index:index + 1].detach().clone()
        row["dataset_names"] = [names[index]]
        rows.append({"feature": context["feature"][index:index + 1].detach().clone(), "inputs": row})
    return rows


def batch_diffusion_context(contexts, *, pad_token_id=0):
    """Re-pad reusable observation contexts for one flow-model invocation.

    This is tensor assembly only: vision/proprioception encoders do not rerun.
    It supports cache entries created in different batches and decoded in any
    order, including repeated entries, with independent external noise rows.
    """
    contexts = list(contexts)
    if not contexts:
        raise ValueError("At least one conditioning context is required")
    rows = [context["inputs"] for context in contexts]
    first = rows[0]
    for row in rows:
        _check_context_keys(row)
        if any(row[key] != first[key] for key in _CONTEXT_SCALARS):
            raise ValueError("Cannot batch contexts with different flow settings")
        if row["inputs_embeds"].shape[0] != 1:
            raise ValueError("Each conditioning context must describe one observation")
        for key in _CONTEXT_SEQUENCES + ("position_ids", "dof_mask"):
            value, reference = row[key], first[key]
            if value is None or reference is None:
                raise ValueError(f"Context {key} is required for batching")
            if value.dtype != reference.dtype or value.device != reference.device:
                raise ValueError(f"Context {key} dtype/device mismatch")
        if row["inputs_embeds"].shape[-1] != first["inputs_embeds"].shape[-1]:
            raise ValueError("Context embedding width mismatch")
        if row["dof_mask"].shape != first["dof_mask"].shape:
            raise ValueError("Context action mask shape mismatch")
    length = max(row["input_ids"].shape[1] for row in rows)
    inputs = {key: first[key] for key in _CONTEXT_SCALARS}
    for key in _CONTEXT_SEQUENCES:
        padded = []
        for row in rows:
            value = row[key]
            pad = length - value.shape[1]
            padding = (0, 0, pad, 0) if key == "inputs_embeds" else (pad, 0)
            padded.append(torch.nn.functional.pad(value, padding, value=pad_token_id if key == "input_ids" else 0))
        inputs[key] = torch.cat(padded, dim=0)
    inputs["position_ids"] = torch.cat([
        torch.nn.functional.pad(row["position_ids"], (length - row["position_ids"].shape[-1], 0), value=1)
        for row in rows
    ], dim=1)
    inputs["dof_mask"] = torch.cat([row["dof_mask"] for row in rows], dim=0)
    inputs["dataset_names"] = [name for row in rows for name in _dataset_names(row["dataset_names"], 1)]
    return {"inputs": inputs, "feature": torch.cat([context["feature"] for context in contexts], dim=0)}


def _check_context_keys(inputs):
    if set(inputs) != _CONTEXT_KEYS:
        raise ValueError(f"Unsupported conditioning context keys: {set(inputs) ^ _CONTEXT_KEYS}")


def _dataset_names(names, batch_size):
    if isinstance(names, str):
        names = [names]
    if len(names) != batch_size or any(not isinstance(name, str) for name in names):
        raise ValueError("Context requires one dataset name per observation")
    return names
