"""CPU regression tests; no pretrained downloads or Transformers installation.

Execute the actual model definitions with small external-backbone substitutes.
Mask math, action MLPs, framework forwarding, dtypes and gradients use real
PyTorch. These tests do not replace the full pretrained GPU training smoke test.

Run: python -m unittest discover -s tests -p test_piper_model_masks.py -v
"""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]


class Config(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error


def config():
    return Config(framework=Config(
        action_model=Config(action_model_type="DiT-B", hidden_size=8, action_dim=9, state_dim=9,
            action_horizon=7, future_action_window_size=6, past_action_window_size=0,
            num_inference_timesteps=2, num_target_vision_tokens=4, add_pos_embed=True,
            max_seq_len=16, noise_beta_alpha=1.5, noise_beta_beta=1., noise_s=.999,
            num_timestep_buckets=1000, diffusion_model_cfg=Config(cross_attention_dim=4)),
        vj2_model=Config(base_encoder="test", num_frames=8, depth=1, num_heads=1,
            special_action_token="<|action_{}|>", num_action_tokens_per_timestep=8,
            embodied_action_token="<|embodied_action|>", num_embodied_action_tokens_per_instruction=32)),
        trainer=Config(repeated_diffusion_steps=3),
        datasets=Config(vla_data=Config(CoT_prompt="{instruction} {actions} {e_actions}"),
                        video_data=Config(CoT_prompt="{instruction} {actions}")))


def definitions(relative_path, names, namespace):
    """Load unchanged function/class bodies while avoiding heavyweight imports."""
    tree = ast.parse((ROOT / relative_path).read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            node = deepcopy(node)
            if isinstance(node, ast.ClassDef):
                node.decorator_list = []
            selected.append(node)
    assert {node.name for node in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(ROOT / relative_path), "exec"), namespace)
    return namespace


class TinyDiT(nn.Module):
    def __init__(self, input_embedding_dim, cross_attention_dim, **kwargs):
        super().__init__()
        self.projection = nn.Linear(input_embedding_dim, 8)
        self.condition = nn.Linear(cross_attention_dim, 8)

    def forward(self, hidden_states, encoder_hidden_states, timestep, **kwargs):
        return self.projection(hidden_states) + self.condition(encoder_hidden_states.mean(1)).unsqueeze(1)


action_namespace = dict(torch=torch, nn=nn, F=F, Beta=torch.distributions.Beta,
                        BatchFeature=dict, DiT=TinyDiT,
                        DiTConfig={"DiT-B": {"input_embedding_dim": 8}})
definitions("starVLA/model/modules/action_model/flow_matching_head/action_encoder.py",
            ["swish", "SinusoidalPositionalEncoding"], action_namespace)
definitions("starVLA/model/modules/action_model/GR00T_ActionHeader.py",
            ["MLP", "ActionEncoder", "masked_action_mse", "FlowmatchingActionHead"], action_namespace)
Head = action_namespace["FlowmatchingActionHead"]
action_loss = action_namespace["masked_action_mse"]


class Tokenizer:
    def __init__(self):
        self.vocab = {"base": 0}

    def get_vocab(self):
        return self.vocab

    def __len__(self):
        return len(self.vocab)

    def add_tokens(self, tokens, special_tokens=True):
        for token in tokens:
            self.vocab[token] = len(self.vocab)
        return len(tokens)

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]


class TinyQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.config = SimpleNamespace(hidden_size=4)
        self.model.embedding = nn.Embedding(64, 4)
        self.model.get_input_embeddings = lambda: self.model.embedding
        self.processor = SimpleNamespace(tokenizer=Tokenizer())
        self.bfloat16_output = False

    def build_qwenvl_inputs(self, images, instructions, **kwargs):
        ids = [self.processor.tokenizer.convert_tokens_to_ids(f"<|action_{i}|>")
               for i in range(3) for _ in range(8)]
        ids += [self.processor.tokenizer.convert_tokens_to_ids("<|embodied_action|>")] * 32
        return {"input_ids": torch.tensor([ids] * len(images))}

    def forward(self, input_ids, **kwargs):
        hidden = self.model.embedding(input_ids)
        if self.bfloat16_output:
            hidden = hidden.bfloat16()
        return SimpleNamespace(hidden_states=[hidden])


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(tubelet_size=2, image_size=16, hidden_size=2)
        self.projection = nn.Linear(1, 2)
        self.dropout = nn.Dropout(.8)
        self.grad_enabled_at_forward = None

    @property
    def device(self):
        return self.projection.weight.device

    def get_vision_features(self, pixel_values_videos):
        self.grad_enabled_at_forward = torch.is_grad_enabled()
        x = pixel_values_videos.float().mean((2, 3, 4)).reshape(-1, 4, 2).mean(-1, keepdim=True)
        return self.dropout(self.projection(x))


class TinyProcessor:
    def __call__(self, videos, **kwargs):
        assert kwargs.get("do_sample_frames") is False
        return {"pixel_values_videos": torch.as_tensor(videos).float().unsqueeze(0)}


class TinyPredictor(nn.Module):
    def __init__(self, embed_dim, action_embed_dim, **kwargs):
        super().__init__()
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.actions = nn.Linear(action_embed_dim, embed_dim)

    def forward(self, states, actions):
        actions = actions.reshape(states.shape[0], 3, 8, -1).mean(2)
        return self.projection(states) + self.actions(actions)


framework_namespace = dict(torch=torch, nn=nn, F=F, np=np, List=List, Optional=Optional, Tuple=Tuple,
    Image=SimpleNamespace(Image=object), AutoTokenizer=object, baseframework=nn.Module,
    logger=SimpleNamespace(info=lambda *args: None, warning=lambda *args: None),
    AutoModel=SimpleNamespace(from_pretrained=lambda *args: TinyEncoder(), from_config=lambda *args: TinyEncoder()),
    AutoConfig=SimpleNamespace(from_pretrained=lambda *args: SimpleNamespace()),
    AutoVideoProcessor=SimpleNamespace(from_pretrained=lambda *args: TinyProcessor()),
    VisionTransformerPredictorAC=TinyPredictor,
    get_vlm_model=lambda **kwargs: TinyQwen(), get_action_model=lambda config: Head(config),
    resize_images=lambda images, **kwargs: images)
definitions("starVLA/model/framework/VLA_JEPA.py",
            ["stack_padding_mask", "masked_world_model_l1", "VLA_JEPA"], framework_namespace)
Framework = framework_namespace["VLA_JEPA"]
world_loss = framework_namespace["masked_world_model_l1"]


def examples():
    rng = np.random.default_rng(8)
    return [dict(image=[object(), object()], lang="Press 24 floor.",
                 video=rng.integers(0, 255, (2, 8, 2, 2, 3), dtype=np.uint8),
                 action=rng.normal(size=(7, 9)).astype(np.float32),
                 state=rng.normal(size=(1, 9)).astype(np.float32)) for _ in range(2)]


class MaskLossTests(unittest.TestCase):
    def test_action_mask_uses_valid_elements_and_zeroes_padded_gradients(self):
        pred = torch.arange(14.).reshape(2, 7, 1).expand(-1, -1, 9).clone().requires_grad_()
        mask = torch.ones(2, 7, dtype=torch.bool)
        mask[0, 0] = False
        mask[1, :3] = False
        loss = action_loss(pred, torch.zeros_like(pred), mask)
        self.assertEqual(loss.item(), (0 + 49 + 64 + 81) / 4)
        loss.backward()
        self.assertTrue(torch.equal(pred.grad[mask], torch.zeros_like(pred.grad[mask])))

    def test_unmasked_loss_matches_original_and_all_padding_has_finite_zero_gradient(self):
        pred = torch.randn(2, 7, 9, requires_grad=True)
        target = torch.randn_like(pred)
        self.assertTrue(torch.equal(action_loss(pred, target), F.mse_loss(pred, target)))
        loss = action_loss(pred, target, torch.ones(2, 7, dtype=torch.bool))
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertTrue(torch.equal(pred.grad, torch.zeros_like(pred)))

    def test_video_mask_skips_first_tubelet_and_expands_spatial_tokens(self):
        pred = torch.tensor([1., 2., 100., 100., 100., 100.]).reshape(1, 6, 1).requires_grad_()
        mask = torch.tensor([[False] * 4 + [True] * 4])
        loss = world_loss(pred, torch.zeros_like(pred), mask, tubelet_size=2)
        self.assertEqual(loss.item(), 1.5)
        loss.backward()
        self.assertTrue(torch.equal(pred.grad[0, 2:], torch.zeros_like(pred.grad[0, 2:])))

    def test_partial_future_tubelet_is_masked_and_all_padding_is_differentiable(self):
        pred = torch.ones(1, 6, 2, requires_grad=True)
        mask = torch.tensor([[False, False, False, True, True, True, True, True]])
        loss = world_loss(pred, torch.zeros_like(pred), mask)
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertTrue(torch.equal(pred.grad, torch.zeros_like(pred)))

    def test_video_no_mask_or_all_real_matches_original_l1(self):
        pred, target = torch.randn(2, 6, 4), torch.randn(2, 6, 4)
        self.assertTrue(torch.equal(world_loss(pred, target), F.l1_loss(pred, target)))
        self.assertTrue(torch.equal(world_loss(pred, target, torch.zeros(2, 8, dtype=torch.bool)), F.l1_loss(pred, target)))

    def test_invalid_masks_fail_instead_of_broadcasting(self):
        with self.assertRaises(ValueError):
            action_loss(torch.ones(2, 7, 9), torch.zeros(2, 7, 9), torch.zeros(2, 8, dtype=torch.bool))
        with self.assertRaises(ValueError):
            world_loss(torch.ones(2, 6, 4), torch.zeros(2, 6, 4), torch.zeros(2, 8))


class ModelIntegrationTests(unittest.TestCase):
    def test_encoder_config_only_initialization_is_explicit_and_default_keeps_pretrained(self):
        auto_model = framework_namespace["AutoModel"]
        auto_config = framework_namespace["AutoConfig"]
        with patch.object(auto_model, "from_pretrained", wraps=auto_model.from_pretrained) as pretrained, \
             patch.object(auto_model, "from_config", wraps=auto_model.from_config) as from_config, \
             patch.object(auto_config, "from_pretrained", wraps=auto_config.from_pretrained) as load_config:
            Framework(config())
            pretrained.assert_called_once_with("test")
            from_config.assert_not_called()
            load_config.assert_not_called()
            cfg = config()
            cfg.framework.vj2_model.init_from_config = True
            model = Framework(cfg)
            pretrained.assert_called_once_with("test")
            load_config.assert_called_once_with("test")
            from_config.assert_called_once()
            self.assertTrue(all(not p.requires_grad for p in model.vj_encoder.parameters()))

    def test_nine_dimensional_head_forward_backward_and_prediction_with_state(self):
        head = Head(config())
        loss = head(torch.randn(2, 4, 4).bfloat16(), torch.randn(2, 7, 9), torch.randn(2, 9))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(head.state_encoder.layer1.weight.grad)
        output = head.predict_action(torch.randn(2, 4, 4).bfloat16(), torch.randn(2, 1, 9).bfloat16())
        self.assertEqual(tuple(output.shape), (2, 7, 9))
        self.assertTrue(torch.isfinite(output).all())

    def test_horizon_disagreement_fails_early(self):
        cfg = config()
        cfg.framework.action_model.action_horizon = 8
        with self.assertRaisesRegex(ValueError, "action_horizon"):
            Head(cfg)

    def test_framework_masks_repeat_in_matching_batch_order_and_encoder_stays_frozen(self):
        model = Framework(config())
        self.assertFalse(model.vj_encoder.training)
        self.assertTrue(all(not p.requires_grad for p in model.vj_encoder.parameters()))
        model.train()
        self.assertFalse(model.vj_encoder.training)
        model.vj_encoder.train()  # Even an external accidental toggle is corrected on use.
        batch = examples()
        batch[0].update(action_is_pad=np.array([False] * 3 + [True] * 4),
                        video_is_pad=np.array([False] * 4 + [True] * 4))
        batch[1].update(action_is_pad=np.array([False] * 6 + [True]),
                        video_is_pad=np.array([False] * 7 + [True]))
        captured = {}
        def capture(module, args, kwargs):
            captured.update(mask=kwargs["action_is_pad"], state=args[2])
        model.action_model.register_forward_pre_hook(capture, with_kwargs=True)
        result = model(batch)
        expected = torch.as_tensor(np.stack([row["action_is_pad"] for row in batch])).repeat(3, 1)
        self.assertTrue(torch.equal(captured["mask"], expected))
        self.assertEqual(tuple(captured["state"].shape), (6, 1, 9))
        self.assertFalse(model.vj_encoder.training)
        self.assertFalse(model.vj_encoder.grad_enabled_at_forward)
        sum(result.values()).backward()
        self.assertTrue(all(p.grad is None for p in model.vj_encoder.parameters()))
        self.assertIsNotNone(model.vj_predictor.projection.weight.grad)

    def test_no_masks_and_all_real_masks_preserve_losses(self):
        model = Framework(config())
        batch = examples()
        torch.manual_seed(11)
        original = model(batch)
        for row in batch:
            row.update(action_is_pad=np.zeros(7, dtype=bool), video_is_pad=np.zeros(8, dtype=bool))
        torch.manual_seed(11)
        masked = model(batch)
        for key in original:
            torch.testing.assert_close(masked[key], original[key])

    def test_predict_action_supports_nine_dimensional_state_without_normalization_stats(self):
        model = Framework(config())
        model.qwen_vl_interface.bfloat16_output = True
        self.assertFalse(hasattr(model, "norm_stats"))
        for shape in ((9,), (1, 9), (1, 1, 9)):
            result = model.predict_action([[object(), object()]], ["Press 24 floor."],
                                          state=np.zeros(shape, dtype=np.float32))
            self.assertEqual(result["normalized_actions"].shape, (1, 7, 9))
            self.assertEqual(result["normalized_actions"].dtype, np.float32)
            self.assertTrue(np.isfinite(result["normalized_actions"]).all())

    def test_incomplete_batch_masks_are_rejected(self):
        batch = examples()
        batch[0]["video_is_pad"] = np.zeros(8, dtype=bool)
        with self.assertRaisesRegex(ValueError, "every example"):
            Framework(config())(batch)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
