"""Reuse the invariant causal prefix within one diffusion solve.

The first Euler step executes the original complete transformer. Later steps
reuse its prefix keys/values and compute action queries only, with rows sliced
from the original eager attention mask. No cache survives a decode request.
"""
import torch


class ReadOnlyPrefixKV:
    """Attention-compatible prefix storage that never appends denoising steps."""

    def __init__(self, cache, prefix_length):
        if prefix_length < 1 or not cache.key_cache or len(cache.key_cache) != len(cache.value_cache):
            raise ValueError("Prefix caching requires nonempty, matched layer KV tensors")
        self.prefix_length = prefix_length
        self.keys = tuple(value[:, :, :prefix_length].detach().clone() for value in cache.key_cache)
        self.values = tuple(value[:, :, :prefix_length].detach().clone() for value in cache.value_cache)

    def update(self, keys, values, layer_idx, cache_kwargs=None):
        return (torch.cat((self.keys[layer_idx], keys), dim=-2),
                torch.cat((self.values[layer_idx], values), dim=-2))


class DiffusionPrefix:
    """Validated eager suffix execution with immutable per-layer prefix KV."""

    def __init__(self, backbone, *, input_ids, inputs_embeds, attention_mask,
                 position_ids, moe_token_types, action_mask, horizon, mode="suffix"):
        if mode not in ("suffix", "full_attention"):
            raise ValueError("flow_prefix_cache_mode must be suffix or full_attention")
        if backbone.config._attn_implementation != "eager":
            raise ValueError("Diffusion prefix caching requires the eager attention mask")
        batch, length = input_ids.shape
        prefix_length = length - horizon
        if prefix_length < 1 or action_mask[:, :prefix_length].any() or not action_mask[:, prefix_length:].all():
            raise ValueError("Diffusion prefix caching requires contiguous final action tokens")
        if (moe_token_types[:, :prefix_length] == 1).any() or not (moe_token_types[:, prefix_length:] == 1).all():
            raise ValueError("Prefix tokens must be causal and action tokens must use expert one")
        if position_ids is None or tuple(position_ids.shape) != (3, batch, length):
            raise ValueError("Diffusion prefix caching requires explicit three-axis RoPE")
        self.backbone = backbone
        self.mode = mode
        self.prefix_length = prefix_length
        self.horizon = horizon
        full_mask = backbone._update_causal_mask(
            attention_mask, inputs_embeds, torch.arange(length, device=inputs_embeds.device),
            None, False, moe_token_types,
        )
        self.attention_mask = full_mask[:, :, prefix_length:, :].contiguous()
        self.full_attention_mask = full_mask
        self.full_position_ids = position_ids
        self.full_cache_position = torch.arange(length, device=inputs_embeds.device)
        self.full_positions = backbone.rotary_emb(inputs_embeds, position_ids) if mode == "full_attention" else None
        self.position_ids = position_ids[:, :, prefix_length:]
        self.cache_position = torch.arange(prefix_length, length, device=inputs_embeds.device)
        self.token_types = moe_token_types[:, prefix_length:].contiguous()
        counts = torch.zeros(backbone.config.num_experts, dtype=torch.long)
        counts[1] = batch * horizon
        self.starts = counts.cumsum(0) - counts
        self.ends = counts.cumsum(0)
        self.cache = None
        self.ready = False
        self.prefix_normalized = {}

    def capture_normalized_prefix(self, layer_index, module, args, kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        self.prefix_normalized[layer_index] = hidden[:, :self.prefix_length].detach().clone()

    def capture_hooks(self):
        """Temporary first-step hooks, owned and removed by the caller."""
        if self.mode != "full_attention":
            return []
        from functools import partial
        return [layer.self_attn.register_forward_pre_hook(
            partial(self.capture_normalized_prefix, index), with_kwargs=True,
        ) for index, layer in enumerate(self.backbone.layers)]

    def capture(self, cache):
        if self.ready:
            raise RuntimeError("A diffusion prefix may only be captured once")
        if self.mode == "full_attention":
            if len(self.prefix_normalized) != len(self.backbone.layers):
                raise RuntimeError("First-step attention hooks did not capture every prefix layer")
        else:
            self.cache = ReadOnlyPrefixKV(cache, self.prefix_length)
        self.ready = True

    def forward(self, action_embeddings):
        if not self.ready:
            raise RuntimeError("Run one complete flow step before cached suffix execution")
        if self.mode == "full_attention":
            return self._forward_full_attention(action_embeddings)
        hidden = action_embeddings.reshape(self.token_types.shape[0], self.horizon, -1)
        positions = self.backbone.rotary_emb(hidden, self.position_ids)
        for layer in self.backbone.layers:
            hidden = layer(
                hidden, attention_mask=self.attention_mask, position_ids=self.position_ids,
                past_key_value=self.cache, token_types=self.token_types,
                start_indices=self.starts, end_indices=self.ends,
                use_cache=False, output_attentions=False,
                cache_position=self.cache_position, position_embeddings=positions,
            )[0]
        return self.backbone.norm(hidden)

    def _forward_full_attention(self, action_embeddings):
        """Keep attention GEMM shapes, while skipping invariant prefix MLPs."""
        hidden = action_embeddings.reshape(self.token_types.shape[0], self.horizon, -1)
        for index, layer in enumerate(self.backbone.layers):
            residual = hidden
            normalized = torch.cat((self.prefix_normalized[index], layer.input_layernorm(hidden)), dim=1)
            attended = layer.self_attn(
                hidden_states=normalized, attention_mask=self.full_attention_mask,
                position_ids=self.full_position_ids, past_key_value=None,
                output_attentions=False, use_cache=False,
                cache_position=self.full_cache_position, position_embeddings=self.full_positions,
            )[0][:, self.prefix_length:]
            hidden = residual + attended
            residual = hidden
            normalized = layer.post_attention_layernorm(hidden)
            if layer.mlp is None:
                hidden = layer.moe(normalized, self.token_types, self.starts, self.ends)
            else:
                hidden = layer.mlp(normalized)
            hidden = residual + hidden
        return self.backbone.norm(hidden)
