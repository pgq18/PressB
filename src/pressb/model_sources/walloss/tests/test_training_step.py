"""Real tiny flow-training forward/backward/update, without checkpoints or a GPU.

Synthetic token/state/action inputs isolate the bundled training model from the
external LeRobot dataset dependency. This does not claim to train the PAP model.
Only the upstream Beta-distribution constructor's hardcoded CUDA placement is
redirected to CPU; transformer, normalization, flow loss and AdamW are real.
"""
import tempfile
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import torch

from wall_x.model.qwen2_5_based.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction


def tiny_train_model():
    config = Qwen2_5_VLConfig(
        vocab_size=32, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        num_experts=2, mlp_moe=False, attention_dropout=0.0,
        experts=[{"hidden_size": 32, "intermediate_size": 48, "hidden_act": "silu"}] * 2,
        rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
        dof_config={"joint": 20}, agent_pos_config={"joint": 20}, noise_scheduler={},
        image_token_id=5, video_token_id=6, vision_start_token_id=4,
        vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32,
                       "num_heads": 4, "in_channels": 3, "patch_size": 2,
                       "spatial_merge_size": 2, "temporal_patch_size": 1,
                       "out_hidden_size": 32, "window_size": 4, "fullatt_block_indexes": [0]},
    )
    config._attn_implementation = "eager"
    processor = types.SimpleNamespace(tokenizer=types.SimpleNamespace(
        convert_tokens_to_ids=lambda token: {"<|action|>": 7, "<|propri|>": 8}[token]
    ))
    tensor_to = torch.Tensor.to

    def cpu_beta(tensor, *args, **kwargs):
        if args == ("cuda",):
            return tensor
        return tensor_to(tensor, *args, **kwargs)

    statistics = {"tiny": {"joint": {"min": [0.0] * 20, "delta": [1.0] * 20}}}
    with patch("torch.Tensor.to", cpu_beta), patch("wall_x.model.action_head.action_statistic_dof", statistics):
        return Qwen2_5_VLMoEForAction(config, processor=processor)


class TrainingStepTest(unittest.TestCase):
    def test_real_flow_backward_updates_backbone_and_action_head_and_roundtrips(self):
        torch.manual_seed(91)
        model = tiny_train_model().train()
        ids = torch.tensor([[2, 8, 7, 7], [3, 8, 7, 7]])
        batch = {
            "input_ids": ids, "attention_mask": torch.ones_like(ids),
            "position_ids": torch.arange(4)[None, None].expand(3, 2, -1),
            "moe_token_types": ids.eq(7).long(), "dataset_names": ["tiny", "tiny"],
            "proprioception": torch.rand(2, 1, 20), "agent_pos_mask": torch.ones(2, 1, 20),
            "action_chunk": torch.randn(2, 2, 20), "dof_mask": torch.ones(2, 2, 20),
        }
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        output = model(mode="train", **batch)
        self.assertTrue(torch.isfinite(output.loss))
        self.assertTrue(torch.isfinite(output.flow_loss))
        output.loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
        optimizer.step()
        changed = [name for name, value in model.named_parameters() if not torch.equal(before[name], value)]
        self.assertTrue(any(name.startswith("model.layers.") for name in changed))
        self.assertIn("action_preprocessor.action_proj_back.weight", changed)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "training.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            restored = tiny_train_model()
            restored.load_state_dict(saved["model"], strict=True)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            restored_optimizer.load_state_dict(saved["optimizer"])
            for name, value in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
            self.assertEqual(len(restored_optimizer.state), len(optimizer.state))


if __name__ == "__main__":
    unittest.main()
