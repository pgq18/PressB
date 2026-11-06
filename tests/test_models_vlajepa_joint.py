"""Native joint contracts, SI conversion and split inference without weights."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from pressb.models.vlajepa_joint import (
    FIXED_GRIPPER_WIDTH_M, JointVLAJEPABackend, MODEL_SCALE, SI_TO_MODEL,
    read_checkpoint_metadata, validate_joint_schema,
)
from pressb.runtime.contract import encode_image, fingerprint
from pressb.runtime.inference import InferenceService


def normalization(size):
    def spec(n):
        value = dict(kind=f"fixed_joint{n}_scale", version=1,
                     feature_names=[f"joint_{i}.pos" for i in range(1, 7)] + (["gripper.pos"] if n == 7 else []),
                     input_units=["degree"] * 6 + (["cm"] if n == 7 else []),
                     scale=MODEL_SCALE[:n].tolist(), offset=[0.] * n, clip=False)
        if n == 7:
            value["gripper_representation"] = "full_width"
        return value
    return spec(7) if size == 7 else dict(kind="separate_joint7_state_joint6_action", version=1,
        state=spec(7), action=spec(6), fixed_gripper_full_width_m=FIXED_GRIPPER_WIDTH_M)


def schema(size):
    result = dict(schema=f"vlajepa.table_buttons.joint{size}.v1", state_dim=7, action_dim=size,
                  action_horizon=7, video_horizon=8, fps=30, cameras=["up", "wrist"],
                  dataset_units=["deg"] * 6 + ["cm"], normalization=normalization(size))
    if size == 6:
        result["fixed_gripper_full_width_m"] = FIXED_GRIPPER_WIDTH_M
    return result


class TinyEncoder:
    def __init__(self):
        self.model = SimpleNamespace(config=SimpleNamespace(hidden_size=3))
        self.calls = 0

    def build_qwenvl_inputs(self, *, images, instructions, prompt_replace_dict):
        self.images = images
        return {"input_ids": torch.tensor([[0, 2, 2, 0]] * len(images))}

    def __call__(self, input_ids, **kwargs):
        self.calls += 1
        # Two embodied tokens average to [4, 5, 6]; other text is excluded.
        hidden = torch.tensor([[[99., 99., 99.], [1., 2., 3.], [7., 8., 9.], [99., 99., 99.]]])
        return SimpleNamespace(hidden_states=[hidden.expand(input_ids.shape[0], -1, -1)])


class TinyHead:
    action_horizon = 7
    dtype = torch.float32
    num_inference_timesteps = 4
    num_timestep_buckets = 1000

    def __init__(self, size):
        self.action_dim = size
        self.config = SimpleNamespace(state_dim=7, add_pos_embed=False)
        self.future_tokens = SimpleNamespace(weight=torch.zeros(1, size))
        self.decode_calls = 0

    def _encode_state(self, state, batch):
        self.last_state = state.clone()
        return state[..., :self.action_dim]

    def action_encoder(self, action, timestep):
        return action

    def model(self, hidden_states, encoder_hidden_states, timestep):
        self.decode_calls += 1
        return torch.full_like(hidden_states, .25)

    def action_decoder(self, output):
        return output


class TinyModel:
    embodied_action_token_id = 2
    replace_prompt, embodied_replace_prompt = "action", "embodied"

    def __init__(self, size):
        self.action_model = TinyHead(size)
        self.qwen_vl_interface = TinyEncoder()
        self.config = SimpleNamespace(framework=SimpleNamespace(vj2_model=SimpleNamespace(
            num_embodied_action_tokens_per_instruction=2)))

    def parameters(self):
        return iter(())

    def eval(self):
        return self


def backend(size):
    policy = SimpleNamespace(model=TinyModel(size), schema=schema(size), normalization=normalization(size),
        device=torch.device("cpu"), resolution=224, allowed_instructions=["Press the red button."],
        identity=dict(model_id=f"tiny_joint{size}", model_sha256="a" * 64, checkpoint_step=10,
                      source_revision="synthetic", config_sha256="b" * 64))
    return JointVLAJEPABackend(policy)


def observation():
    image = encode_image(np.zeros((480, 640, 3), dtype=np.uint8))
    return dict(task="Press the red button.", state=[np.pi / 2] * 6 + [.04], images=dict(up=image, wrist=image))


@pytest.mark.parametrize("size", [6, 7])
def test_native_dimensions_si_normalization_features_and_external_noise(size):
    model = backend(size)
    metadata = model.describe()
    assert metadata["noise"]["shape"] == [7, size]
    assert metadata["action"]["names"] == [f"joint{i}" for i in range(1, 7)] + (["gripper_width"] if size == 7 else [])
    assert metadata["action"]["units"] == ["rad"] * 6 + (["m"] if size == 7 else [])
    assert len(metadata["observation"]["state"]["names"]) == 7
    rng = torch.random.get_rng_state().clone()
    (context, feature), = model.encode_batch([observation()], [11])
    assert torch.equal(rng, torch.random.get_rng_state())
    np.testing.assert_array_equal(feature, [4, 5, 6])
    np.testing.assert_allclose(context.state.numpy(), .5)
    assert [image.size for image in model.model.qwen_vl_interface.images[0]] == [(224, 224)] * 2
    noise = np.full((7, size), .75, dtype=np.float32)
    actions, = model.decode_batch([context], [noise])
    # Four Euler steps integrate constant velocity .25 for total time 1.
    np.testing.assert_allclose(actions[:, :6], np.pi, rtol=1e-6)
    if size == 7:
        np.testing.assert_allclose(actions[:, 6], .08)
    else:
        assert metadata["fixed_gripper_width_m"] == FIXED_GRIPPER_WIDTH_M
    assert model.model.qwen_vl_interface.calls == 1
    np.testing.assert_array_equal(noise, .75)
    assert model.model.action_model.decode_calls == 4


@pytest.mark.parametrize("size", [6, 7])
def test_baseline_seed_batch_independence_and_cached_context_reuse(size):
    model = backend(size)
    batch = model.encode_batch([observation(), observation()], [4, 99])
    single = model.encode_batch([observation()], [99])
    contexts = [entry[0] for entry in batch]
    baseline = model.decode_batch(contexts, [None, None])
    model.decode_batch(contexts, [np.zeros((7, size)), np.ones((7, size))])
    repeated = model.decode_batch(contexts, [None, None])
    np.testing.assert_array_equal(baseline, repeated)
    np.testing.assert_array_equal(baseline[1], model.decode_batch([single[0][0]], [None])[0])
    for context, seed in zip(contexts, [4, 99]):
        expected = torch.randn((1, 7, size), generator=torch.Generator().manual_seed(seed))
        torch.testing.assert_close(context.baseline_noise, expected, rtol=0, atol=0)


def test_invalid_observation_and_noise_fail_before_model_work():
    model = backend(6)
    bad = observation()
    bad["task"] = "Press 31 floor."
    with pytest.raises(ValueError, match="trained"):
        model.encode_batch([observation(), bad], [1, 2])
    assert model.model.qwen_vl_interface.calls == 0
    context = model.encode_batch([observation()], [1])[0][0]
    for noise in (np.zeros((7, 7)), np.full((7, 6), np.nan), np.zeros((7, 6), dtype=int)):
        with pytest.raises(ValueError):
            model.decode_batch([context, context], [None, noise])
    assert model.model.action_model.decode_calls == 0
    for size in (6, 8, 9):
        bad = observation()
        bad["state"] = [0.] * size
        with pytest.raises(ValueError):
            model.encode_batch([bad], [1])


def test_runtime_microbatch_context_and_noise_contract():
    model = backend(6)
    service = InferenceService(model, batch_size=2)
    encoded = service.encode(dict(observations=[observation()] * 3, seeds=[1, 2, 3], include_base_actions=True))
    assert model.model.qwen_vl_interface.calls == 2
    items = encoded["items"]
    decoded = service.decode(dict(items=[dict(context_id=item["context_id"], initial_noise=np.zeros((7, 6)).tolist()) for item in reversed(items)]))
    assert np.shape(decoded["items"][0]["actions"]) == (7, 6)
    assert decoded["items"][0]["actions"] != items[-1]["base_actions"]


@pytest.mark.parametrize("mutate", [
    lambda s: s.update(action_dim=9), lambda s: s.update(state_dim=6),
    lambda s: s.update(cameras=["global", "wrist"]),
    lambda s: s["normalization"]["state"].update(input_units=["rad"] * 6 + ["m"]),
    lambda s: s["normalization"]["action"].update(scale=[1.] * 6),
    lambda s: s["normalization"]["state"].update(gripper_representation="half_width"),
    lambda s: s["normalization"].update(fixed_gripper_full_width_m=.0002),
])
def test_joint_contract_rejects_incompatible_semantics(mutate):
    spec = schema(6)
    mutate(spec)
    with pytest.raises(ValueError):
        validate_joint_schema(spec, spec["normalization"])


def test_checkpoint_metadata_binds_schema_normalization_and_training_config(tmp_path):
    run = tmp_path / "run"
    checkpoint = run / "checkpoints/best"
    checkpoint.mkdir(parents=True)
    (run / "processor").mkdir()
    (checkpoint / "model.pt").write_bytes(b"unit-test-weights")
    spec = schema(6)
    spec["dataset_group"] = "near_red"
    spec["dataset_sha256"] = "a" * 64
    split = dict(source_export_manifest_sha256=spec["dataset_sha256"],
                 camera_keys=["observation.images.up", "observation.images.wrist"])
    spec["split_sha256"] = fingerprint(split)
    framework = dict(qwenvl=dict(init_from_config=True), vj2_model=dict(init_from_config=True),
                     action_model=dict(state_dim=7, action_dim=6, action_horizon=7))
    training = dict(seed=42, framework=framework, data=dict(resolution_size=224), trainer={},
                    normalization=spec["normalization"])
    spec["training_fingerprint"] = fingerprint(training)
    config = dict(run_id="near_red", seed=42, framework=framework,
                  datasets=dict(vla_data=training["data"]), trainer={})
    manifest = dict(complete=True, schema=spec["schema"], step=1, model_bytes=len(b"unit-test-weights"),
                    **{k: spec[k] for k in ("dataset_sha256", "split_sha256", "training_fingerprint")})
    for path, value in (("action_schema.json", spec), ("normalization.json", spec["normalization"]),
                        ("split.json", split), ("training_contract.json", dict(contract=training, fingerprint=fingerprint(training))),
                        ("checkpoints/best/manifest.json", manifest)):
        (run / path).write_text(json.dumps(value))
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    result = read_checkpoint_metadata(run)
    assert result["action_dim"] == 6
    assert result["allowed_instructions"] == ["Press the red button."]
    config["framework"]["action_model"]["action_dim"] = 7
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="training contract"):
        read_checkpoint_metadata(run)


def test_legacy_imports_alias_actual_implementation_and_pose_rejects_joints():
    from pressb.models import vlajepa, walloss
    from pressb.rl_nodes import vlajepa as legacy_vlajepa, walloss as legacy_walloss
    assert legacy_vlajepa is vlajepa
    assert legacy_walloss is walloss
    with pytest.raises(ValueError, match="nine-dimensional"):
        vlajepa.VLAJEPABackend(SimpleNamespace(action_dim=7, action_horizon=7), contract_profile="pose9")


def test_pose_adapter_has_explicit_configurable_instruction_profile():
    from pressb.models.vlajepa import VLAJEPABackend
    frozen = SimpleNamespace(action_dim=9, action_horizon=7, feature_dim=3,
        health=lambda: dict(checkpoint_sha256="a" * 64, checkpoint_step=1, frozen=True,
                            controller_gripper_width_m=.008),
        state_preparer=lambda value: value)
    model = VLAJEPABackend(frozen, contract_profile="pose9", instruction_pattern=r"Reach .+\.")
    image = encode_image(np.zeros((2, 2, 3), dtype=np.uint8))
    obs = dict(task="Reach the marker.", state=[0., 0., 0., 1., 0., 0., 0., .008],
               images=dict(global_=image, wrist=image))
    obs["images"]["global"] = obs["images"].pop("global_")
    model.validate_observations([obs])
    obs["task"] = "Press 31 floor."
    with pytest.raises(ValueError, match="pattern"):
        model.validate_observations([obs])
    with pytest.raises(ValueError, match="retains"):
        VLAJEPABackend(frozen, instruction_pattern=r"Reach .+\.")
