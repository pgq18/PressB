"""CPU checks for native batched preprocessing and independently reusable caches."""
import copy
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import BatchFeature

from wall_x.model.flow_context import batch_diffusion_context, split_diffusion_context
from wall_x.serving.policy.utils import prepare_batch
from wall_x.serving.policy.wall_x_policy import WallXPolicy
from test_flow_context import tiny_model


def prepared_inputs(lengths=(4, 6)):
    length = max(lengths)
    ids = torch.zeros((len(lengths), length), dtype=torch.long)
    masks = torch.zeros_like(ids)
    positions = torch.ones((3, len(lengths), length), dtype=torch.long)
    for index, count in enumerate(lengths):
        ids[index, -count:] = torch.tensor([5] + [index + 2] * (count - 3) + [7, 7])
        masks[index, -count:] = 1
        positions[:, index, -count:] = torch.arange(count)[None]
    return dict(input_ids=ids, attention_mask=masks, position_ids=positions,
                pixel_values=torch.arange(len(lengths), dtype=torch.float32)[:, None] + 2,
                image_grid_thw=torch.ones((len(lengths), 3), dtype=torch.long),
                moe_token_types=ids.eq(7), dof_mask=torch.ones((len(lengths), 2, 4)),
                dataset_names=[f"dataset_{index}" for index in range(len(lengths))])


class BatchVisual:
    spatial_merge_unit = 1
    dtype = torch.float32

    def __init__(self):
        self.calls = 0

    def __call__(self, values, grid_thw, segmented_attention=False, sample_token_counts=None):
        self.calls += 1
        self.segmented_attention = segmented_attention
        return values.expand(-1, 8).clone()


class PaddingAwareBackbone:
    def __init__(self):
        self.batch_sizes = []
        self.embed_calls = 0

    def embed_tokens(self, ids):
        self.embed_calls += 1
        return ids.float().unsqueeze(-1).expand(-1, -1, 8).clone() / 10

    def __call__(self, **kwargs):
        value = kwargs["inputs_embeds"]
        self.batch_sizes.append(len(value))
        valid = kwargs["attention_mask"].bool() & ~kwargs["moe_token_types"].bool()
        weights = valid.float().unsqueeze(-1)
        prefix = (value * weights).sum(1, keepdim=True) / weights.sum(1, keepdim=True)
        position = kwargs["position_ids"][0].float().unsqueeze(-1) / 100
        return types.SimpleNamespace(last_hidden_state=value + prefix + position)


def policy():
    value = WallXPolicy.__new__(WallXPolicy)
    value.model = tiny_model()
    value.model.model = PaddingAwareBackbone()
    value.model.visual = BatchVisual()
    value.model.config._attn_implementation = "eager"
    value.predict_mode = "diffusion"
    value.fixed_action_dim = 4
    value.action_dim = 3
    value.pred_horizon = 2
    value.processor = types.SimpleNamespace(tokenizer=types.SimpleNamespace(pad_token_id=0))
    return value


class NativeBatchTest(unittest.TestCase):
    def test_batch_encode_and_decode_are_single_model_invocations(self):
        value = policy()
        value.prepare_observations = lambda observations: prepared_inputs()
        contexts = value.encode_batch([{}, {}], num_inference_timesteps=3)
        self.assertEqual(value.model.visual.calls, 1)
        self.assertTrue(value.model.visual.segmented_attention)
        self.assertEqual(value.model.model.embed_calls, 1)
        self.assertEqual([row["inputs"]["input_ids"].shape[1] for row in contexts], [4, 6])
        noise = torch.randn((2, 2, 4))
        output = value.decode_batch(contexts, noise)["action"]
        self.assertEqual(output.shape, (2, 2, 3))
        self.assertEqual(value.model.model.batch_sizes, [2, 2, 2])
        self.assertEqual(value.model.visual.calls, 1)
        expected = np.concatenate([
            value.decode(context, noise[index:index + 1])["action"]
            for index, context in enumerate(contexts)
        ])
        np.testing.assert_allclose(output, expected, rtol=0, atol=0)

    def test_context_reorder_repeat_and_noise_changes_are_isolated(self):
        value = policy()
        value.prepare_observations = lambda observations: prepared_inputs()
        contexts = value.encode_batch([{}, {}], num_inference_timesteps=3)
        before = copy.deepcopy(contexts)
        noise = torch.randn((3, 2, 4))
        reordered = [contexts[1], contexts[0], contexts[1]]
        output = value.decode_batch(reordered, noise)["action"]
        expected = np.concatenate([
            value.decode(context, noise[index:index + 1])["action"]
            for index, context in enumerate(reordered)
        ])
        np.testing.assert_allclose(output, expected, rtol=0, atol=0)
        changed = noise.clone()
        changed[0] += 1
        different = value.decode_batch(reordered, changed)["action"]
        self.assertFalse(np.array_equal(output[0], different[0]))
        np.testing.assert_array_equal(output[1:], different[1:])
        for original, current in zip(before, contexts):
            for key, tensor in original["inputs"].items():
                if isinstance(tensor, torch.Tensor):
                    torch.testing.assert_close(tensor, current["inputs"][key], rtol=0, atol=0)

    def test_split_clones_storage_and_preserves_positions_and_features(self):
        value = policy()
        original = value.model.prepare_diffusion_context(
            **prepared_inputs(), pred_horizon=2, action_dim=4, num_inference_timesteps=3,
        )
        contexts = split_diffusion_context(original)
        assembled = batch_diffusion_context(contexts)
        for key in ("input_ids", "attention_mask", "position_ids", "moe_token_types", "dof_mask"):
            torch.testing.assert_close(assembled["inputs"][key], original["inputs"][key])
        torch.testing.assert_close(assembled["feature"], original["feature"])
        for context in contexts:
            tensor = context["inputs"]["inputs_embeds"]
            self.assertEqual(tensor.untyped_storage().nbytes(), tensor.numel() * tensor.element_size())
        contexts[0]["inputs"]["inputs_embeds"].zero_()
        self.assertTrue(original["inputs"]["inputs_embeds"].count_nonzero())
        self.assertTrue(contexts[1]["inputs"]["inputs_embeds"].count_nonzero())

    def test_reject_incompatible_or_interior_padding_context(self):
        value = policy()
        original = value.model.prepare_diffusion_context(
            **prepared_inputs(), pred_horizon=2, action_dim=4, num_inference_timesteps=3,
        )
        contexts = split_diffusion_context(original)
        contexts[1]["inputs"]["num_inference_timesteps"] = 2
        with self.assertRaisesRegex(ValueError, "flow settings"):
            batch_diffusion_context(contexts)
        original["inputs"]["attention_mask"][0] = torch.tensor([0, 1, 0, 1, 1, 1])
        with self.assertRaisesRegex(ValueError, "left padding"):
            split_diffusion_context(original)
        value.model.config._attn_implementation = "flash_attention_2"
        with self.assertRaisesRegex(ValueError, "eager"):
            value.decode_batch(contexts)


class PreprocessingBatchTest(unittest.TestCase):
    def test_preprocessor_once_preserves_camera_order_states_and_dataset_names(self):
        captured = []
        def preprocess(**kwargs):
            captured.append(kwargs)
            return BatchFeature({"input_ids": torch.tensor([[0, 2, 7, 7], [3, 2, 7, 7]]),
                                 "attention_mask": torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])})
        processor = types.SimpleNamespace(tokenizer=types.SimpleNamespace(
            padding_side="left", convert_tokens_to_ids=lambda token: 7,
        ))
        observations = [dict(prompt="red", state=np.array([1, 2]), dataset_names=["a"],
                             up=np.full((28, 28, 3), 11, np.uint8), wrist=np.full((28, 28, 3), 12, np.uint8)),
                        dict(prompt="a longer blue instruction", state=[3, 4], dataset_names="b",
                             up=np.full((28, 28, 3), 21, np.uint8), wrist=np.full((28, 28, 3), 22, np.uint8))]
        with patch("wall_x.serving.policy.utils.preprocesser_call", side_effect=preprocess):
            batch = prepare_batch(observations, processor, ["up", "wrist"], 2, 2, 2, 4,
                                  128, 28, 28 * 28, 28 * 28, "diffusion", "cpu")
        self.assertEqual(len(captured), 1)
        self.assertEqual([[np.asarray(img)[0, 0, 0] for img in row] for row in captured[0]["images"]],
                         [[11, 12], [21, 22]])
        self.assertIn("Instruction: red", captured[0]["text"][0])
        self.assertIn("Instruction: a longer blue instruction", captured[0]["text"][1])
        torch.testing.assert_close(batch.proprioception[:, 0], torch.tensor([[1., 2., 0., 0.], [3., 4., 0., 0.]]))
        self.assertEqual(batch.dataset_names, ["a", "b"])
        self.assertEqual(tuple(batch.dof_mask.shape), (2, 2, 4))
        self.assertFalse(batch.agent_pos_mask[:, :, 2:].any())

    def test_empty_batch_and_truncated_action_tokens_fail_early(self):
        processor = types.SimpleNamespace(tokenizer=types.SimpleNamespace(
            padding_side="left", convert_tokens_to_ids=lambda token: 7,
        ))
        args = (processor, [], 2, 2, 2, 4, 128, 28, 784, 784, "diffusion", "cpu")
        with self.assertRaisesRegex(ValueError, "At least one"):
            prepare_batch([], *args)
        with patch("wall_x.serving.policy.utils.preprocesser_call", return_value=BatchFeature({
            "input_ids": torch.tensor([[2, 7]]), "attention_mask": torch.ones((1, 2)),
        })):
            with self.assertRaisesRegex(ValueError, "truncation"):
                prepare_batch(dict(prompt="red", state=[1, 2], dataset_names=["a"]), *args)


if __name__ == "__main__":
    unittest.main()
