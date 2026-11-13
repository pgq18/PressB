import json
from types import SimpleNamespace

import numpy as np
import pytest

from pressb.models import pi05
from pressb.runtime.contract import encode_image
from pressb.runtime.inference import InferenceService


def policy_config():
    return SimpleNamespace(
        robot_state_feature=SimpleNamespace(shape=(2,)),
        action_feature=SimpleNamespace(shape=(2,)),
        image_features={
            "observation.images.external": None,
            "observation.images.wrist": None,
            "observation.images.empty_camera_0": None,
        },
        chunk_size=4,
        max_state_dim=32,
        max_action_dim=32,
        image_resolution=(8, 8),
        rtc_config=None,
        num_inference_steps=10,
        validate_features=lambda: None,
    )


def io_contract():
    return {
        "format": "pressb.pi05.v1",
        "checkpoint_step": 7,
        "lerobot_version": "0.4.4",
        "observation": {"state": {"names": ["joint1", "gripper_width"], "units": ["rad", "m"]},
                        "cameras": ["up", "wrist"]},
        "camera_keys": {"up": "observation.images.external", "wrist": "observation.images.wrist"},
        "action": {"representation": "joint_position", "frame": "robot_base",
                   "names": ["joint1", "gripper_width"], "units": ["rad", "m"], "control_hz": 30},
        "state_scale": [2., 100.],
        "action_scale": [.5, .01],
    }


def test_pi05_contract_uses_padded_noise_and_unpadded_wire_actions():
    metadata, cameras, state_scale, action_scale = pi05._wire_contract(io_contract(), policy_config(), "pi05-test")
    assert metadata["checkpoint_step"] == 7
    assert metadata["noise"]["shape"] == [4, 32]
    assert metadata["action"]["horizon"] == 4
    assert len(metadata["action"]["names"]) == 2
    assert metadata["feature_dim"] == 2
    assert cameras["up"] == "observation.images.external"
    np.testing.assert_allclose(state_scale, [2., 100.])
    np.testing.assert_allclose(action_scale, [.5, .01])


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"camera_keys": {"up": "observation.images.external"}}, "map each"),
        ({"camera_keys": {"up": "observation.images.empty_camera_0", "wrist": "observation.images.wrist"}}, "cover"),
        ({"state_scale": [0., 1.]}, "nonzero"),
        ({"action_scale": [1.]}, "shape"),
        ({"checkpoint_step": -1}, "nonnegative"),
    ],
)
def test_pi05_contract_rejects_ambiguous_wire_mapping(changes, message):
    with pytest.raises(ValueError, match=message):
        pi05._wire_contract({**io_contract(), **changes}, policy_config(), "pi05-test")


def test_pi05_inference_service_batches_processed_context_and_seeded_noise(tmp_path, monkeypatch):
    pytest.importorskip("lerobot")
    import torch
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def predict_action_chunk(self, batch, *, noise):
            self.batch_sizes.append(len(noise))
            assert tuple(noise.shape[1:]) == (4, 32)
            assert batch["observation.images.external"].shape[1:] == (3, 8, 8)
            return noise[:, :, :2] + batch[OBS_LANGUAGE_TOKENS][:, None, :]

    def preprocessor(batch):
        state = batch[pi05.STATE_KEY]
        return {**batch, pi05.STATE_KEY: state - 1,
                OBS_LANGUAGE_TOKENS: state + torch.tensor([[len(task), 0] for task in batch["task"]]),
                OBS_LANGUAGE_ATTENTION_MASK: torch.ones_like(state, dtype=torch.bool)}

    def postprocessor(actions):
        return actions + 1

    (tmp_path / "config.json").write_text('{"type":"pi05"}')
    (tmp_path / "model.safetensors").write_bytes(b"fake unit test model")
    (tmp_path / pi05.IO_FILENAME).write_text(json.dumps(io_contract()))
    policy = Policy()
    monkeypatch.setattr(pi05, "load_pi05_config", lambda *a, **k: policy_config())
    monkeypatch.setattr(pi05, "load_pi05_policy", lambda *a, **k: policy)
    monkeypatch.setattr(pi05, "_load_processors", lambda *a, **k: (preprocessor, postprocessor, "processor"))

    backend = pi05.create_backend({"checkpoint": str(tmp_path), "device": "cpu"})
    service = InferenceService(backend, batch_size=2)
    rows = [
        {"state": [1, .03], "task": "press", "images": {"up": encode_image(np.ones((10, 8, 3), np.uint8)),
                                                       "wrist": encode_image(np.ones((8, 8, 3), np.uint8))}},
        {"state": [2, .02], "task": "pick", "images": {"up": encode_image(np.ones((8, 10, 3), np.uint8)),
                                                      "wrist": encode_image(np.ones((6, 8, 3), np.uint8))}},
    ]
    encoded = service.encode({"observations": rows, "seeds": [12, 17], "include_base_actions": True})["items"]
    np.testing.assert_allclose([item["features"] for item in encoded], [[1., 2.], [3., 1.]])
    # A subsequent request must not alter the earlier encoded contexts or
    # consume randomness that changes their baseline actions.
    service.encode({"observations": rows[:1], "seeds": [19]})
    noise = np.zeros((4, 32), dtype=np.float32)
    decoded = service.decode({"items": [
        {"context_id": encoded[1]["context_id"], "initial_noise": noise},
        {"context_id": encoded[0]["context_id"]},
    ]})["items"]
    np.testing.assert_allclose(decoded[0]["actions"], [[4.5, .03]] * 4)
    np.testing.assert_array_equal(decoded[1]["actions"], encoded[0]["base_actions"])
    assert policy.batch_sizes == [2, 2]
    assert service.health()["feature_kind"] == "normalized_proprioception"
    assert service.release({"context_ids": [item["context_id"] for item in encoded]})["released"] == 2
    service.close()


def test_pi05_camera_padding_matches_native_policy_for_mixed_frame_sizes():
    pytest.importorskip("lerobot")
    import torch
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    key = "observation.images.up"
    config = SimpleNamespace(image_features={key: None}, image_resolution=(8, 8))
    dummy = SimpleNamespace(config=config, parameters=lambda: iter([torch.zeros(1)]))
    backend = pi05.Pi05Backend.__new__(pi05.Pi05Backend)
    backend.device = torch.device("cpu")
    backend.image_resolution = config.image_resolution
    rng = np.random.default_rng(42)
    frames = [rng.integers(0, 256, (12, 8, 3), dtype=np.uint8), rng.integers(0, 256, (7, 10, 3), dtype=np.uint8)]
    rows = [{"images": {"up": encode_image(frame)}} for frame in frames]
    adapter_frames = backend._camera_batch(rows, "up")
    images, _ = PI05Policy._preprocess_images(dummy, {key: adapter_frames})
    for index, frame in enumerate(frames):
        raw = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().div_(255.)
        expected, _ = PI05Policy._preprocess_images(dummy, {key: raw})
        torch.testing.assert_close(images[0][index], expected[0][0], rtol=0, atol=0)


def test_pi05_loader_rejects_incomplete_native_weights(tmp_path, monkeypatch):
    pytest.importorskip("lerobot")
    import torch
    from lerobot.policies.pi05 import modeling_pi05
    from safetensors.torch import save_file, save_model

    class Config:
        pass

    class Policy(torch.nn.Module):
        config_class = Config

        def __init__(self, config):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2))
            self.alias = self.weight

    monkeypatch.setattr(modeling_pi05, "PI05Policy", Policy)
    weights = tmp_path / "model.safetensors"
    original = Policy(Config())
    torch.nn.init.constant_(original.weight, 2.)
    save_model(original, weights)
    restored = pi05.load_pi05_policy(tmp_path, Config())
    torch.testing.assert_close(restored.weight, original.weight)
    assert restored.alias is restored.weight
    save_file({"unrelated": torch.zeros(2)}, weights)
    with pytest.raises(RuntimeError, match="Missing key"):
        pi05.load_pi05_policy(tmp_path, Config())


def test_pi05_legacy_base_remap_uses_tied_parameters_and_loads_strictly(tmp_path, monkeypatch):
    pytest.importorskip("lerobot")
    import torch
    from lerobot.policies.pi05 import modeling_pi05
    from safetensors.torch import save_file

    class Config:
        pass

    class Policy(torch.nn.Module):
        config_class = Config

        def __init__(self, config):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embedding = torch.nn.Parameter(torch.zeros(2))
            self.model.lm_head = self.model.embedding
            self.model.other = torch.nn.Parameter(torch.zeros(2))

        def _fix_pytorch_state_dict_keys(self, weights, config):
            return weights

    monkeypatch.setattr(modeling_pi05, "PI05Policy", Policy)
    path = tmp_path / "model.safetensors"
    save_file({"lm_head": torch.full((2,), 2.), "other": torch.full((2,), 3.)}, path)
    policy = pi05.load_pi05_policy(tmp_path, Config(), pretrained_base=True)
    torch.testing.assert_close(policy.model.embedding, torch.full((2,), 2.))
    save_file({"lm_head": torch.full((2,), 2.)}, path)
    with pytest.raises(RuntimeError, match="model.other"):
        pi05.load_pi05_policy(tmp_path, Config(), pretrained_base=True)
