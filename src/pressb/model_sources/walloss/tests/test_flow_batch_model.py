"""CPU checks of batched routing and the checkpoint's actual eager mask.

No weights or GPU allocation are needed. The tiny backbone uses the real mask
builder and attention over every token, so padding or action cross-talk changes
the resulting actions instead of being hidden by a constant mock output.
"""
import types
import unittest
from unittest.mock import patch

import torch

from test_flow_context import tiny_model
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEModel


class TinyEagerBackbone:
    def __init__(self):
        self.config = types.SimpleNamespace(_attn_implementation="eager", sliding_window=None)
        self.training = False
        self.calls = 0
        self._prepare_4d_causal_attention_mask_with_cache_position = (
            Qwen2_5_VLMoEModel._prepare_4d_causal_attention_mask_with_cache_position
        )
        self._update_causal_mask = types.MethodType(Qwen2_5_VLMoEModel._update_causal_mask, self)

    def embed_tokens(self, ids):
        return torch.sin(ids.float().unsqueeze(-1) / torch.arange(1, 9))

    def __call__(self, **kwargs):
        self.calls += 1
        value = kwargs["inputs_embeds"]
        mask = self._update_causal_mask(
            kwargs["attention_mask"], value, torch.arange(value.shape[1]), None,
            False, kwargs["moe_token_types"],
        )[:, 0]
        weights = (value @ value.transpose(1, 2) / 8 ** 0.5 + mask).softmax(-1)
        return types.SimpleNamespace(last_hidden_state=value + weights @ value)


def model():
    result = tiny_model()
    result.model = TinyEagerBackbone()
    result.config._attn_implementation = "eager"
    return result


def inputs(rows):
    width = max(map(len, rows))
    ids = torch.tensor([[0] * (width - len(row)) + row for row in rows])
    attention = ids.ne(0)
    positions = attention.long().cumsum(-1) - 1
    positions.masked_fill_(~attention, 1)
    return dict(
        input_ids=ids, attention_mask=attention,
        position_ids=positions[None].expand(3, -1, -1),
        moe_token_types=ids.eq(7).long(),
        dof_mask=torch.ones((len(rows), 2, 4)), pred_horizon=2, action_dim=4,
        dataset_names=["tiny"] * len(rows), num_inference_timesteps=3,
    )


class FlowBatchModelTest(unittest.TestCase):
    def test_visual_sample_boundaries_count_images_and_videos_per_observation(self):
        class CapturingVisual:
            dtype = torch.float32
            spatial_merge_unit = 4

            def __init__(self):
                self.calls = []

            def __call__(self, values, grid_thw, segmented_attention=False, sample_token_counts=None):
                self.calls.append((segmented_attention, sample_token_counts))
                return values

        policy = model()
        policy.config.video_token_id = 12
        policy.visual = CapturingVisual()
        data = inputs([[5, 12, 12, 2, 7, 7], [5, 5, 12, 9, 7, 7]])
        data.update(pixel_values=torch.ones((3, 8)), image_grid_thw=torch.tensor([[1, 2, 2]] * 3),
                    pixel_values_videos=torch.ones((3, 8)), video_grid_thw=torch.tensor([[1, 2, 2]] * 3))
        policy.prepare_diffusion_context(**data)
        self.assertEqual(len(policy.visual.calls), 2)
        self.assertTrue(all(segmented for segmented, _ in policy.visual.calls))
        torch.testing.assert_close(policy.visual.calls[0][1], torch.tensor([4, 8]))
        torch.testing.assert_close(policy.visual.calls[1][1], torch.tensor([8, 4]))

        singleton = inputs([[5, 2, 7, 7]])
        singleton.update(pixel_values=torch.ones((1, 8)), image_grid_thw=torch.tensor([[1, 2, 2]]))
        policy.prepare_diffusion_context(**singleton)
        self.assertEqual(policy.visual.calls[-1], (False, None))

    def test_padded_batch_matches_single_actions_features_and_preserves_context(self):
        policy = model()
        rows = [[2, 3, 7, 7], [4, 6, 9, 7, 7]]
        noise = torch.arange(16).reshape(2, 2, 4).float() / 13
        context = policy.prepare_diffusion_context(**inputs(rows))
        saved = context["inputs"]["inputs_embeds"].clone()
        combined = policy.decode_diffusion_context(context, noise)["predict_action"]
        self.assertEqual(policy.model.calls, 3)  # One B=2 call per ODE step.
        for index, row in enumerate(rows):
            single = policy.prepare_diffusion_context(**inputs([row]))
            result = policy.decode_diffusion_context(single, noise[index:index + 1])["predict_action"]
            torch.testing.assert_close(combined[index:index + 1], result, atol=2e-6, rtol=1e-6)
            torch.testing.assert_close(context["feature"][index], single["feature"][0])
        torch.testing.assert_close(saved, context["inputs"]["inputs_embeds"], atol=0, rtol=0)
        changed_noise = noise.clone()
        changed_noise[0] += 1
        changed = policy.decode_diffusion_context(context, changed_noise)["predict_action"]
        torch.testing.assert_close(changed[1], combined[1], atol=0, rtol=0)
        self.assertFalse(torch.equal(changed[0], combined[0]))

    def test_per_row_action_counts_reject_equal_total_with_wrong_row_assignment(self):
        # The total still equals B*H; only per-row validation catches this.
        with self.assertRaisesRegex(ValueError, "Each diffusion batch row"):
            model().prepare_diffusion_context(**inputs([[2, 3, 7], [4, 7, 7, 7]]))

    def test_multimodal_fused_rope_left_padding_is_removed_from_each_row(self):
        policy = model()
        policy.rope_deltas = None
        policy.config.vision_config = types.SimpleNamespace(spatial_merge_size=2, tokens_per_second=2)
        policy.config.video_token_id = 12
        policy.config.vision_start_token_id = 11
        data = inputs([[2, 3, 7, 7], [4, 6, 9, 7, 7]])
        expected = data["position_ids"].clone()
        data["position_ids"] = None
        # Embeddings are already available, so the CPU test needs no vision
        # encoder. A 1x1 merged image grid has ordinary sequential positions.
        data["inputs_embeds"] = policy.model.embed_tokens(data["input_ids"])
        data["image_grid_thw"] = torch.tensor([[1, 2, 2], [1, 2, 2]])

        def fused_positions(**kwargs):
            ids = kwargs["input_ids"]
            mask = kwargs["attention_mask"]
            positions = torch.arange(ids.shape[1])[None, None].expand(3, *ids.shape).clone()
            positions.masked_fill_(~mask.bool().unsqueeze(0), 1)
            return positions, torch.zeros((ids.shape[0], 1), dtype=torch.long)

        module = "wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act.ops.get_rope_index"
        with patch(module, side_effect=fused_positions):
            context = policy.prepare_diffusion_context(**data)
        torch.testing.assert_close(context["inputs"]["position_ids"], expected, atol=0, rtol=0)
        torch.testing.assert_close(policy.rope_deltas, torch.tensor([[-1], [0]]), atol=0, rtol=0)

        # The kernel's text-only branch already discounts padding.
        data.pop("image_grid_thw")
        with patch(module, return_value=(expected, torch.tensor([[-1], [0]]))):
            text_context = policy.prepare_diffusion_context(**data)
        torch.testing.assert_close(text_context["inputs"]["position_ids"], expected, atol=0, rtol=0)

    def test_padded_action_tokens_and_unsupported_attention_fail_before_encoding(self):
        data = inputs([[2, 3, 7, 7]])
        data["attention_mask"][0, -1] = False
        with self.assertRaisesRegex(ValueError, "action tokens cannot be padding"):
            model().prepare_diffusion_context(**data)
        for implementation in ("sdpa", "flash_attention_2"):
            policy = model()
            policy.config._attn_implementation = implementation
            with self.assertRaisesRegex(ValueError, "require eager attention"):
                policy.prepare_diffusion_context(**inputs([[2, 7, 7], [2, 3, 7, 7]]))

    def test_rtc_uses_each_rows_history_and_matches_single_calls(self):
        policy = model()
        rows = [[2, 3, 7, 7], [4, 6, 9, 7, 7]]
        noise = torch.arange(16).reshape(2, 2, 4).float() / 13
        history = torch.stack([torch.full((2, 4), -0.5), torch.full((2, 4), 0.5)])
        options = dict(predict_mode="diffusion", rtc_s=0, rtc_d=1, rtc_beta=0.2)
        combined = policy.predict(
            **options, **inputs(rows), initial_noise=noise, prev_action=history,
        )["predict_action"]
        for index, row in enumerate(rows):
            result = policy.predict(
                **options, **inputs([row]), initial_noise=noise[index:index + 1],
                prev_action=history[index],  # Legacy 2-D singleton input.
            )["predict_action"]
            torch.testing.assert_close(combined[index:index + 1], result, atol=2e-6, rtol=1e-6)
        with self.assertRaisesRegex(ValueError, "history for every batch row"):
            policy.predict(**options, **inputs(rows), initial_noise=noise, prev_action=history[0])


if __name__ == "__main__":
    unittest.main()
