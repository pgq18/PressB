"""CPU equivalence and isolation checks for memory-linear visual batching."""
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from wall_x.model.vision_batch import vision_attention_segments, segmented_vision_attention, VisionAttentionGroup
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl import (
    Qwen2_5_VLVisionAttention, Qwen2_5_VLVisionSdpaAttention,
    Qwen2_5_VisionTransformerPretrainedModel,
)

from wall_x.model.qwen2_5_based.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig


class VisionBatchTest(unittest.TestCase):
    def test_dense_and_segmented_layers_match_for_unequal_image_windows(self):
        torch.manual_seed(12)
        bounds = torch.tensor([0, 3, 8, 11, 11, 16], dtype=torch.int32)
        segments = vision_attention_segments(bounds, token_count=16)
        self.assertEqual([tuple(item.shape) for item in segments], [(2, 3), (2, 5)])
        hidden = torch.randn((16, 16))
        phase = torch.randn((16, 4))
        embeddings = (phase.cos(), phase.sin())
        for layer_type in (Qwen2_5_VLVisionAttention, Qwen2_5_VLVisionSdpaAttention):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(layer=layer_type.__name__, dtype=dtype):
                    layer = layer_type(16, num_heads=4).to(dtype)
                    dense = layer(hidden.to(dtype), bounds, position_embeddings=embeddings)
                    batch = layer(hidden.to(dtype), bounds, position_embeddings=embeddings,
                                  attention_segments=segments)
                    torch.testing.assert_close(batch, dense, rtol=1e-5 if dtype == torch.float32 else 0.01,
                                               atol=1e-6 if dtype == torch.float32 else 0.004)

    def test_no_cross_image_or_window_leakage(self):
        torch.manual_seed(15)
        bounds = torch.tensor([0, 4, 8, 12], dtype=torch.int32)
        segments = vision_attention_segments(bounds, token_count=12)
        q, k, v = (torch.randn(12, 2, 4) for _ in range(3))
        for implementation in ("eager", "sdpa"):
            before = segmented_vision_attention(q, k, v, segments, implementation=implementation)
            changed = v.clone()
            changed[4:8] += 100
            after = segmented_vision_attention(q, k, changed, segments, implementation=implementation)
            torch.testing.assert_close(after[:4], before[:4], rtol=0, atol=0)
            torch.testing.assert_close(after[8:], before[8:], rtol=0, atol=0)
            self.assertGreater(float((after[4:8] - before[4:8]).abs().min()), 99)

    def test_sdpa_processes_equal_length_windows_in_true_batch(self):
        bounds = torch.arange(0, 65, 4, dtype=torch.int32)
        segments = vision_attention_segments(bounds, token_count=64)
        q = torch.randn(64, 2, 4)
        calls = []
        original = F.scaled_dot_product_attention
        def record(query, key, value, *args, **kwargs):
            calls.append(tuple(query.shape))
            return original(query, key, value, *args, **kwargs)
        with patch("wall_x.model.vision_batch.F.scaled_dot_product_attention", side_effect=record):
            output = segmented_vision_attention(q, q, q, segments, implementation="sdpa")
        self.assertEqual(calls, [(16, 2, 4, 4)])
        self.assertEqual(output.shape, q.shape)

    def test_eager_does_not_allocate_global_token_attention_matrix(self):
        bounds = torch.arange(0, 65, 4, dtype=torch.int32)
        segments = vision_attention_segments(bounds, token_count=64)
        q = torch.randn(64, 2, 4)
        shapes = []
        original = torch.matmul
        def record(left, right):
            output = original(left, right)
            shapes.append(tuple(output.shape))
            return output
        with patch("wall_x.model.vision_batch.torch.matmul", side_effect=record):
            segmented_vision_attention(q, q, q, segments, implementation="eager")
        self.assertEqual(shapes, [(16, 2, 4, 4), (16, 2, 4, 4)])

    def test_full_visual_encoder_keeps_window_order_and_default_path(self):
        config = Qwen2_5_VLVisionConfig(
            depth=2, hidden_size=16, intermediate_size=32, num_heads=4,
            in_channels=3, patch_size=2, spatial_merge_size=2, temporal_patch_size=1,
            out_hidden_size=8, window_size=4, fullatt_block_indexes=[1],
        )
        config._attn_implementation = "sdpa"
        model = Qwen2_5_VisionTransformerPretrainedModel(config).eval()
        grid = torch.tensor([[1, 4, 4], [1, 4, 6], [1, 4, 4]])
        pixels = torch.randn((56, 12))
        module = "wall_x.model.qwen2_5_based.modeling_qwen2_5_vl"
        def windows(**kwargs):
            index, cu = model.get_window_index(kwargs["grid_thw"])
            return index, torch.tensor(cu, dtype=torch.int32)
        with patch(module + ".ops.rot_pos_emb", side_effect=lambda frequency, shape, merge: model.rot_pos_emb(shape)), \
             patch(module + ".ops.get_window_index", side_effect=windows), \
             patch(module + ".segmented_vision_attention", wraps=segmented_vision_attention) as segmented:
            with torch.no_grad():
                original = model(pixels, grid)
                self.assertEqual(segmented.call_count, 0)
                batched = model(pixels, grid, segmented_attention=True)
                self.assertEqual(segmented.call_count, 2)
                observation_batch = model(pixels, grid, segmented_attention=True,
                                          sample_token_counts=[16, 40])
                observation_singles = torch.cat([model(pixels[:16], grid[:1]), model(pixels[16:], grid[1:])])
            self.assertEqual(segmented.call_count, 4)
        torch.testing.assert_close(batched, original, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(observation_batch, observation_singles, rtol=1e-5, atol=1e-6)

    def test_observation_groups_preserve_original_internal_masks_and_widths(self):
        boundaries = torch.tensor([0, 3, 8, 11, 16, 18, 20], dtype=torch.int32)
        groups = vision_attention_segments(boundaries, token_count=20, sample_token_counts=[8, 8, 4])
        self.assertEqual([tuple(group.indices.shape) for group in groups], [(2, 8), (1, 4)])
        self.assertEqual([tuple(group.mask.shape) for group in groups], [(2, 1, 8, 8), (1, 1, 4, 4)])
        self.assertTrue(groups[0].mask[0, 0, :3, :3].all())
        self.assertFalse(groups[0].mask[0, 0, :3, 3:].any())
        self.assertTrue(groups[0].mask[0, 0, 3:, 3:].all())
        q, k, v = (torch.randn(20, 2, 4) for _ in range(3))
        for implementation in ("eager", "sdpa"):
            output = segmented_vision_attention(q, k, v, groups, implementation=implementation)
            reference = []
            for start, end, split in ((0, 8, 3), (8, 16, 3), (16, 20, 2)):
                width = end - start
                mask = torch.zeros((1, width, width), dtype=torch.bool)
                mask[:, :split, :split] = mask[:, split:, split:] = True
                # Feed one observation as a block with its original dense mask.
                group = VisionAttentionGroup(torch.arange(start, end)[None], mask[None])
                reference.append(segmented_vision_attention(q, k, v, (group,), implementation=implementation)[start:end])
            torch.testing.assert_close(output, torch.cat(reference), rtol=1e-5, atol=1e-6)

    def test_observation_groups_reject_blocks_crossing_samples(self):
        with self.assertRaisesRegex(ValueError, "crosses"):
            vision_attention_segments(torch.tensor([0, 6, 8]), token_count=8, sample_token_counts=[4, 4])
        with self.assertRaisesRegex(ValueError, "cover"):
            vision_attention_segments(torch.tensor([0, 4, 8]), token_count=8, sample_token_counts=[3, 4])

    def test_invalid_boundaries_fail(self):
        for boundaries in ([1, 4], [0, 3], [0, 4, 2, 4]):
            with self.assertRaises(ValueError):
                vision_attention_segments(torch.tensor(boundaries), token_count=4)


if __name__ == "__main__":
    unittest.main()
