"""Prefix reuse against real eager transformer layers, without checkpoints."""
import unittest
from unittest.mock import patch

import torch

from wall_x.model.flow_prefix import DiffusionPrefix
from wall_x.model.qwen2_5_based.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEModel


def fixture(moe=False):
    torch.manual_seed(81)
    config = Qwen2_5_VLConfig(
        vocab_size=32, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        num_experts=2, mlp_moe=moe, attention_dropout=0.0,
        experts=[{"hidden_size": 32, "intermediate_size": 48, "hidden_act": "silu"}] * 2,
        rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
    )
    config._attn_implementation = "eager"
    backbone = Qwen2_5_VLMoEModel(config).eval()
    ids = torch.tensor([[0, 0, 4, 5, 6, 7, 7], [2, 3, 4, 5, 6, 7, 7]])
    mask = ids.ne(0).long()
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(~mask.bool(), 1)
    return backbone, dict(input_ids=ids, inputs_embeds=backbone.embed_tokens(ids),
                         attention_mask=mask, position_ids=positions[None].expand(3, -1, -1),
                         moe_token_types=ids.eq(7).long(), action_mask=ids.eq(7), horizon=2)


class FlowPrefixTest(unittest.TestCase):
    @torch.no_grad()
    def test_moe_action_expert_routing_matches_full_sequence(self):
        backbone, args = fixture(moe=True)
        prefix = DiffusionPrefix(backbone, **args)
        counts = torch.bincount(args["moe_token_types"].flatten(), minlength=2)
        full_args = {key: args[key] for key in ("inputs_embeds", "attention_mask", "position_ids", "moe_token_types")}
        full_args.update(start_indices=counts.cumsum(0) - counts, end_indices=counts.cumsum(0))

        def permute(hidden, indices):
            order = indices.argsort(stable=True)
            return hidden[order], order

        def unpermute(hidden, order, probabilities):
            output = torch.empty_like(hidden)
            output[order] = hidden
            return output

        module = "wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act.ops"
        with patch(module + ".permute", side_effect=permute), patch(module + ".unpermute", side_effect=unpermute):
            first = backbone(**full_args, use_cache=True)
            prefix.capture(first.past_key_values)
            action = torch.randn(2, 2, 32)
            full_args["inputs_embeds"] = torch.cat((args["inputs_embeds"][:, :-2], action), dim=1)
            full = backbone(**full_args, use_cache=False).last_hidden_state[:, -2:]
            torch.testing.assert_close(prefix.forward(action), full, atol=5e-7, rtol=5e-6)

    @torch.no_grad()
    def test_full_attention_keeps_shapes_and_skips_prefix_mlp(self):
        backbone, args = fixture(moe=True)
        prefix = DiffusionPrefix(backbone, **args, mode="full_attention")
        counts = torch.bincount(args["moe_token_types"].flatten(), minlength=2)
        full_args = {key: args[key] for key in ("inputs_embeds", "attention_mask", "position_ids", "moe_token_types")}
        full_args.update(start_indices=counts.cumsum(0) - counts, end_indices=counts.cumsum(0))
        def permute(hidden, indices):
            order = indices.argsort(stable=True)
            return hidden[order], order
        def unpermute(hidden, order, probabilities):
            output = torch.empty_like(hidden)
            output[order] = hidden
            return output
        module = "wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act.ops"
        with patch(module + ".permute", side_effect=permute), patch(module + ".unpermute", side_effect=unpermute):
            handles = prefix.capture_hooks()
            try:
                first = backbone(**full_args, use_cache=True)
            finally:
                for handle in handles:
                    handle.remove()
            prefix.capture(first.past_key_values)
            attention_shapes, prefix_mlp_calls = [], []
            hooks = []
            for layer in backbone.layers:
                hooks.append(layer.self_attn.register_forward_pre_hook(
                    lambda module, values, kw: attention_shapes.append(list(kw["hidden_states"].shape)), with_kwargs=True))
                hooks.append(layer.moe.experts[0].register_forward_hook(lambda *unused: prefix_mlp_calls.append(1)))
            action = torch.randn(2, 2, 32)
            actual = prefix.forward(action)
            for hook in hooks:
                hook.remove()
            self.assertEqual(attention_shapes, [[2, 7, 32], [2, 7, 32]])
            self.assertEqual(prefix_mlp_calls, [])
            full_args["inputs_embeds"] = torch.cat((args["inputs_embeds"][:, :-2], action), dim=1)
            full = backbone(**full_args, use_cache=False).last_hidden_state[:, -2:]
            torch.testing.assert_close(actual, full, atol=5e-7, rtol=5e-6)

    @torch.no_grad()
    def test_suffix_matches_full_transformer_over_repeated_flow_steps(self):
        backbone, args = fixture()
        prefix = DiffusionPrefix(backbone, **args)
        full_args = {key: args[key] for key in ("inputs_embeds", "attention_mask", "position_ids", "moe_token_types")}
        first = backbone(**full_args, use_cache=True)
        prefix.capture(first.past_key_values)
        saved = [value.clone() for value in prefix.cache.keys + prefix.cache.values]
        # Prefix keys must be independently owned, not views of a full decode.
        for value in first.past_key_values.key_cache:
            value.zero_()
        for _ in range(4):
            action = torch.randn(2, 2, 32)
            full_args["inputs_embeds"] = torch.cat((args["inputs_embeds"][:, :-2], action), dim=1)
            full = backbone(**full_args, use_cache=False).last_hidden_state[:, -2:]
            actual = prefix.forward(action)
            torch.testing.assert_close(actual, full, atol=5e-7, rtol=5e-6)
        for actual, expected in zip(prefix.cache.keys + prefix.cache.values, saved):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_action_bidirectionality_padding_and_shapes_are_preserved(self):
        backbone, args = fixture()
        prefix = DiffusionPrefix(backbone, **args)
        self.assertEqual(tuple(prefix.attention_mask.shape), (2, 1, 2, 7))
        self.assertTrue((prefix.attention_mask[:, :, :, -2:] == 0).all())
        self.assertTrue((prefix.attention_mask[0, :, :, :2] < -1e20).all())
        with self.assertRaisesRegex(RuntimeError, "complete flow step"):
            prefix.forward(torch.randn(2, 2, 32))

    def test_rejects_noncausal_prefix_noncontiguous_actions_and_alternate_attention(self):
        backbone, args = fixture()
        args["action_mask"] = args["action_mask"].roll(1, dims=-1)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            DiffusionPrefix(backbone, **args)
        args["action_mask"] = args["action_mask"].roll(-1, dims=-1)
        args["moe_token_types"][0, 2] = 1
        with self.assertRaisesRegex(ValueError, "causal"):
            DiffusionPrefix(backbone, **args)
        backbone.config._attn_implementation = "sdpa"
        with self.assertRaisesRegex(ValueError, "eager"):
            DiffusionPrefix(backbone, **args)


if __name__ == "__main__":
    unittest.main()
