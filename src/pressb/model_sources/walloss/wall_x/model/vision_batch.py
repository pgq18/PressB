"""Block-diagonal vision attention without a dense cross-image mask.

Image/window boundaries already isolate attention in Wall-X. Grouping equal
length blocks executes those independent blocks as a genuine tensor batch, so
memory scales with the number of images rather than its square.
"""
import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


@dataclass(frozen=True)
class VisionAttentionGroup:
    indices: torch.Tensor
    mask: torch.Tensor


def vision_attention_segments(cu_seqlens, *, token_count, sample_token_counts=None):
    """Build reusable index matrices once per visual forward, not per layer."""
    boundaries = cu_seqlens.detach().cpu().tolist()
    if not boundaries or boundaries[0] != 0 or boundaries[-1] != token_count:
        raise ValueError("Vision attention boundaries must cover all input tokens")
    if any(end < start for start, end in zip(boundaries, boundaries[1:])):
        raise ValueError("Vision attention boundaries must be nondecreasing")
    if sample_token_counts is not None:
        return _sample_attention_groups(boundaries, sample_token_counts, cu_seqlens.device, token_count)
    starts_by_length = {}
    for start, end in zip(boundaries, boundaries[1:]):
        if end < start:
            raise ValueError("Vision attention boundaries must be nondecreasing")
        if end > start:
            starts_by_length.setdefault(end - start, []).append(start)
    return tuple(
        torch.tensor(starts, device=cu_seqlens.device, dtype=torch.long)[:, None]
        + torch.arange(length, device=cu_seqlens.device)[None, :]
        for length, starts in starts_by_length.items()
    )


def _sample_attention_groups(boundaries, sample_token_counts, device, token_count):
    """Preserve each observation's original softmax width and internal mask.

    Shortening BF16 reductions to individual windows changes rounding enough
    to amplify through this checkpoint's visual layers. Keep the legacy token
    extent per observation while batching observations as separate matrices.
    """
    if isinstance(sample_token_counts, torch.Tensor):
        counts = sample_token_counts.detach().cpu().tolist()
    else:
        counts = list(sample_token_counts)
    if any(type(count) is not int or count < 1 for count in counts) or sum(counts) != token_count:
        raise ValueError("Visual sample token counts must be positive and cover all tokens")
    groups = {}
    start = cursor = 0
    for count in counts:
        end = start + count
        mask = torch.zeros((count, count), dtype=torch.bool)
        while cursor + 1 < len(boundaries) and boundaries[cursor] < end:
            left, right = boundaries[cursor:cursor + 2]
            if left < start or right > end:
                raise ValueError("A vision attention block crosses observation boundaries")
            mask[left - start:right - start, left - start:right - start] = True
            cursor += 1
        indices, masks = groups.setdefault(count, ([], []))
        indices.append(start)
        masks.append(mask)
        start = end
    return tuple(VisionAttentionGroup(
        torch.tensor(starts, device=device, dtype=torch.long)[:, None]
        + torch.arange(length, device=device)[None, :],
        torch.stack(masks)[:, None].to(device),
    ) for length, (starts, masks) in groups.items())


def segmented_vision_attention(q, k, v, segments, *, implementation):
    """Compute independent blocks in length groups and restore token order.

    q/k/v are [total_tokens, heads, head_dim]; each segment index tensor is
    [number_of_blocks, block_length]. There is no [total_tokens,total_tokens]
    intermediate. SDPA math matches the legacy 3D masked SDPA path instead of
    implicitly switching to a different CUDA kernel when adding a batch axis.
    """
    if implementation not in {"eager", "sdpa"}:
        raise ValueError("Unsupported segmented vision attention implementation")
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 3:
        raise ValueError("Vision q/k/v must share [tokens, heads, head_dim] shape")
    output = torch.empty_like(q)
    heads, width = q.shape[1:]
    for group in segments:
        indices = group.indices if isinstance(group, VisionAttentionGroup) else group
        mask = group.mask if isinstance(group, VisionAttentionGroup) else None
        block_count, length = indices.shape
        flat = indices.flatten()
        grouped_q, grouped_k, grouped_v = [
            value.index_select(0, flat).reshape(block_count, length, heads, width).transpose(1, 2)
            for value in (q, k, v)
        ]
        if implementation == "eager":
            weights = torch.matmul(grouped_q, grouped_k.transpose(-2, -1)) / math.sqrt(width)
            if mask is not None:
                attention_mask = torch.zeros(mask.shape, dtype=q.dtype, device=q.device)
                attention_mask.masked_fill_(~mask, torch.finfo(q.dtype).min)
                weights = weights + attention_mask
            weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(q.dtype)
            result = torch.matmul(weights, grouped_v)
        else:
            with sdpa_kernel(SDPBackend.MATH):
                result = F.scaled_dot_product_attention(
                    grouped_q, grouped_k, grouped_v, attn_mask=mask, dropout_p=0.0,
                )
        output.index_copy_(0, flat, result.transpose(1, 2).reshape(-1, heads, width))
    return output
