"""CPU tests of the actual flow predict path with tiny synthetic model layers.

Run in the Wall-X environment: python -m unittest discover -s tests -p test_flow_context.py
No checkpoint or GPU allocation is needed.
"""
import types
import unittest

import torch

from wall_x.model.flow_context import initial_flow_noise
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction


class TinyBackbone:
    def __init__(self):
        self.embed_calls = 0

    def embed_tokens(self, ids):
        self.embed_calls += 1
        return ids.float().unsqueeze(-1).expand(-1, -1, 8).clone() / 10

    def __call__(self, **kwargs):
        value = kwargs["inputs_embeds"]
        return types.SimpleNamespace(last_hidden_state=value + value[:, :2].mean(dim=1, keepdim=True))


class TinyVisual:
    dtype = torch.float32

    def __init__(self):
        self.calls = 0

    def __call__(self, values, grid_thw):
        self.calls += 1
        return torch.ones((1, 8)) * values.mean()


def tiny_model():
    model = types.SimpleNamespace(
        model=TinyBackbone(), visual=TinyVisual(),
        config=types.SimpleNamespace(output_attentions=False, output_hidden_states=False,
                                     use_return_dict=True, image_token_id=5, num_experts=2),
        action_token_id_set={"action_token_id": 7},
        action_preprocessor=types.SimpleNamespace(
            step=lambda timestep, noisy_action, dof_mask: torch.nn.functional.pad(noisy_action, (0, 4)),
            action_proj_back=lambda hidden: hidden[..., :4],
            normalizer_action=types.SimpleNamespace(unnormalize_data=lambda action, names: action),
        ),
    )
    for name in ("predict", "prepare_diffusion_context", "decode_diffusion_context"):
        setattr(model, name, types.MethodType(getattr(Qwen2_5_VLMoEForAction, name), model))
    return model


def inputs():
    return dict(input_ids=torch.tensor([[5, 2, 7, 7]]), attention_mask=torch.ones((1, 4)),
                position_ids=torch.zeros((3, 1, 4), dtype=torch.long),
                pixel_values=torch.ones((1, 8)), image_grid_thw=torch.tensor([[1, 1, 1]]),
                moe_token_types=torch.tensor([[0, 0, 1, 1]]),
                dof_mask=torch.tensor([[[1., 1., 0., 0.], [1., 1., 0., 0.]]]),
                pred_horizon=2, action_dim=4, dataset_names=["tiny"], num_inference_timesteps=3)


class FlowContextTest(unittest.TestCase):
    def test_default_noise_preserves_rng_and_external_noise_does_not_consume_it(self):
        reference = torch.zeros(1)
        torch.manual_seed(41)
        expected = torch.randn((1, 2, 4))
        torch.manual_seed(41)
        actual = initial_flow_noise(None, batch_size=1, horizon=2, action_dim=4, reference=reference)
        self.assertTrue(torch.equal(expected, actual))
        rng = torch.get_rng_state().clone()
        actual = initial_flow_noise(expected, batch_size=1, horizon=2, action_dim=4, reference=reference)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        actual.zero_()
        self.assertFalse(torch.equal(actual, expected))

    def test_cached_and_uncached_actions_match_and_encoder_runs_once(self):
        model = tiny_model()
        noise = torch.randn((1, 2, 4))
        direct = model.predict(predict_mode="diffusion", initial_noise=noise, **inputs())["predict_action"]
        context = model.prepare_diffusion_context(**inputs())
        features_before = context["feature"].clone()
        embedded_before = context["inputs"]["inputs_embeds"].clone()
        self.assertEqual(model.visual.calls, 2)
        self.assertEqual(tuple(features_before.shape), (1, 8))
        decoded = model.decode_diffusion_context(context, noise)["predict_action"]
        changed = model.decode_diffusion_context(context, noise + 1)["predict_action"]
        repeated = model.decode_diffusion_context(context, noise)["predict_action"]
        torch.testing.assert_close(direct, decoded, rtol=0, atol=0)
        torch.testing.assert_close(decoded, repeated, rtol=0, atol=0)
        self.assertFalse(torch.equal(changed, decoded))
        self.assertEqual(model.visual.calls, 2)
        self.assertEqual(model.model.embed_calls, 2)
        self.assertTrue(torch.equal(features_before, context["feature"]))
        self.assertTrue(torch.equal(embedded_before, context["inputs"]["inputs_embeds"]))

    def test_invalid_noise_and_nonflow_requests_fail(self):
        model = tiny_model()
        context = model.prepare_diffusion_context(**inputs())
        for noise in (torch.zeros((1, 2, 2)), torch.zeros((1, 2, 4), dtype=torch.long),
                      torch.full((1, 2, 4), float("nan"))):
            with self.assertRaises(ValueError):
                model.decode_diffusion_context(context, noise)
        with self.assertRaises(ValueError):
            model.predict(predict_mode="fast", initial_noise=torch.zeros((1, 2, 4)), **inputs())


if __name__ == "__main__":
    unittest.main()
