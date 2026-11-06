"""CPU-only inference contract tests: no model download or GPU allocation."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pose = load_module("piper_policy_test_pose", ROOT / "starVLA/dataloader/piper_lerobot.py")
policy = load_module("piper_policy_test", ROOT / "starVLA/inference/piper_policy.py")


@pytest.fixture(autouse=True)
def helpers(monkeypatch):
    monkeypatch.setattr(policy, "_pose_helpers", lambda: pose)


def schema():
    return dict(policy._SCHEMA, dataset_sha256="a" * 64, split_sha256="b" * 64, training_fingerprint="c" * 64)


def test_pose9_to_pose8_xyz_rows_quaternion_and_explicit_fixed_gripper():
    raw8 = np.array([.123456789, -.7, .31, np.sqrt(.5), 0, 0, np.sqrt(.5), .081], dtype=np.float32)
    raw9 = pose.pose8_to_pose9(raw8)
    np.testing.assert_allclose(raw9[3:], [0, -1, 0, 1, 0, 0], atol=1e-7)
    actual = policy.pose9_to_pose8(np.tile(raw9, (7, 1)))
    assert actual.dtype == np.float32 and actual.shape == (7, 8)
    np.testing.assert_array_equal(actual[:, :3], np.tile(raw8[:3], (7, 1)))
    np.testing.assert_allclose(pose.quaternion_wxyz_to_matrix(actual[:, 3:7]),
                               np.tile(pose.quaternion_wxyz_to_matrix(raw8[3:7]), (7, 1, 1)), atol=1e-7)
    np.testing.assert_array_equal(actual[:, 7], np.full(7, .008, dtype=np.float32))
    custom = policy.pose9_to_pose8(raw9, controller_gripper_width_m=.025)
    assert custom[7] == np.float32(.025)
    # Predictions need not be orthonormal; project rotation only, keep raw xyz.
    scaled = raw9.copy()
    scaled[3:6] *= 2
    scaled[6:] = raw9[6:] * 3 + raw9[3:6] * .2
    np.testing.assert_allclose(policy.pose9_to_pose8(scaled), actual[0], atol=1e-7)


@pytest.mark.parametrize("bad", [[0, 0, 0, 0, 0, 0], [1, 0, 0, 2, 0, 0], [1, 0, 0, np.nan, 1, 0]])
def test_bad_prediction_rotation_is_not_replaced_with_identity(bad):
    with pytest.raises(ValueError):
        policy.pose9_to_pose8(np.array([.1, .2, .3, *bad]))


@pytest.mark.parametrize("key,value", [("rotation", "first two columns"), ("pose_frame", "world"),
                                         ("normalization", "standard"), ("action_horizon", 8),
                                         ("state_dim", 8), ("gripper", "learned"),
                                         ("fps", 10), ("terminal_action", "measured state")])
def test_schema_rejects_incompatible_semantics(key, value):
    value_schema = schema()
    assert policy.validate_action_schema(value_schema) == value_schema
    value_schema[key] = value
    with pytest.raises(ValueError, match="action schema"):
        policy.validate_action_schema(value_schema)


def test_state8_and_state9_are_identical_without_using_gripper():
    raw = np.array([.12, -.31, .9, 2, 0, 0, 0, .003], dtype=np.float32)
    expected = pose.pose8_to_pose9(raw)
    np.testing.assert_array_equal(policy.prepare_state(raw), expected.reshape(1, 1, 9))
    np.testing.assert_array_equal(policy.prepare_state(expected), expected.reshape(1, 1, 9))
    changed = raw.copy()
    changed[-1] = .07
    np.testing.assert_array_equal(policy.prepare_state(changed), policy.prepare_state(raw))
    for bad in (raw[None], np.zeros(9), np.full(8, np.nan), expected * 2):
        with pytest.raises(ValueError):
            policy.prepare_state(bad)


class SmallModel(torch.nn.Module):
    def __init__(self, tied=False):
        super().__init__()
        for name in ("qwen_vl_interface", "vj_encoder", "vj_predictor", "action_model"):
            setattr(self, name, torch.nn.Linear(2, 2))
        if tied:
            self.qwen_vl_interface.other_weight = self.qwen_vl_interface.weight


def test_weights_load_all_four_components_strictly(tmp_path):
    source, target = SmallModel(), SmallModel()
    path = tmp_path / "model.pt"
    torch.save(source.state_dict(), path)
    assert policy.load_trained_state(target, path) == 8
    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value)
    broken = source.state_dict()
    del broken["action_model.bias"]
    torch.save(broken, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        policy.load_trained_state(target, path)
    torch.save({"state_dict": source.state_dict()}, path)
    with pytest.raises(ValueError, match="plain complete"):
        policy.load_trained_state(target, path)


def test_tied_weights_cannot_be_silently_overwritten(tmp_path):
    source, target = SmallModel(tied=True), SmallModel(tied=True)
    path = tmp_path / "model.pt"
    torch.save(source.state_dict(), path)
    policy.load_trained_state(target, path)
    broken = {key: value.clone() for key, value in source.state_dict().items()}
    broken["qwen_vl_interface.other_weight"] += 1
    torch.save(broken, path)
    with pytest.raises(ValueError, match="tied parameter"):
        policy.load_trained_state(target, path)


def test_load_preserves_trained_fp32_in_bf16_constructor_and_parameter_ties(tmp_path):
    source, target = SmallModel(tied=True), SmallModel(tied=True).to(dtype=torch.bfloat16)
    with torch.no_grad():
        source.qwen_vl_interface.weight.fill_(1.)
        source.qwen_vl_interface.weight[0, 0] = 1. + 1e-5
    assert source.qwen_vl_interface.weight[0, 0].to(torch.bfloat16).float() == 1.
    path = tmp_path / "model.pt"
    torch.save(source.state_dict(), path)
    policy.load_trained_state(target, path)
    assert target.qwen_vl_interface.weight is target.qwen_vl_interface.other_weight
    for name, value in source.state_dict().items():
        assert target.state_dict()[name].dtype == value.dtype
        assert torch.equal(target.state_dict()[name], value)


def checkpoint_fixture(tmp_path):
    run = tmp_path / "run"
    checkpoint = run / "checkpoints/step_000001"
    checkpoint.mkdir(parents=True)
    (run / "processor").mkdir()
    (run / "processor/tokenizer.json").write_text('{"test":true}')
    (run / "config.yaml").write_text("test: true\n")
    (checkpoint / "model.pt").write_bytes(b"fake model used only for checksum tests")
    split = dict(source_export_manifest_sha256="a" * 64, camera_keys=list(policy.CAMERA_KEYS),
                 train_source_episode_ids=[0], val_source_episode_ids=[1])
    contract = dict(config="only integrity is tested here")
    action_schema = dict(schema(), split_sha256=policy.canonical_sha256(split),
                         training_fingerprint=policy.canonical_sha256(contract))
    manifest = dict(complete=True, step=1, model_sha256=policy.sha256(checkpoint / "model.pt"),
                    **{key: action_schema[key] for key in ("dataset_sha256", "split_sha256", "training_fingerprint")})
    for path, value in ((run / "split.json", split), (run / "action_schema.json", action_schema),
                        (run / "training_contract.json", dict(fingerprint=action_schema["training_fingerprint"], contract=contract)),
                        (checkpoint / "manifest.json", manifest)):
        path.write_text(json.dumps(value))
    (run / "checkpoints/best").symlink_to("step_000001", target_is_directory=True)
    return run, checkpoint


def test_checkpoint_hash_schema_split_and_complete_binding(tmp_path):
    run, checkpoint = checkpoint_fixture(tmp_path)
    validated = policy.validate_checkpoint(run / "checkpoints/best")
    assert validated["checkpoint"] == checkpoint.resolve()
    assert validated["provenance"]["processor_files_sha256"]["tokenizer.json"] == policy.sha256(run / "processor/tokenizer.json")
    assert policy.validate_checkpoint(checkpoint / "model.pt")["manifest"] == validated["manifest"]
    (checkpoint / "model.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        policy.validate_checkpoint(checkpoint)


@pytest.mark.parametrize("target,key,value,match", [
    ("manifest", "complete", False, "complete"),
    ("manifest", "step", 0, "training step"),
    ("manifest", "dataset_sha256", "b" * 64, "dataset_sha256 mismatch"),
    ("schema", "pose_frame", "world", "action schema"),
    ("split", "train_source_episode_ids", [1], "Split canonical"),
    ("contract", "fingerprint", "d" * 64, "contract fingerprint"),
])
def test_checkpoint_tampering_fails_closed(tmp_path, target, key, value, match):
    run, checkpoint = checkpoint_fixture(tmp_path)
    path = {"manifest": checkpoint / "manifest.json", "schema": run / "action_schema.json",
            "split": run / "split.json", "contract": run / "training_contract.json"}[target]
    content = json.loads(path.read_text())
    content[key] = value
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match=match):
        policy.validate_checkpoint(checkpoint)


def test_saved_processor_matches_full_vocabulary_and_token_ids():
    def processor(vocab=None, pad=0):
        tokenizer = SimpleNamespace(get_vocab=lambda: vocab or {"a": 0, "<|action_0|>": 1},
                                    pad_token_id=pad, bos_token_id=None, eos_token_id=2, padding_side="right")
        return SimpleNamespace(tokenizer=tokenizer)
    model = SimpleNamespace(qwen_vl_interface=SimpleNamespace(processor=processor()))
    saved = processor()
    policy.install_saved_processor(model, saved)
    assert model.qwen_vl_interface.processor is saved and saved.tokenizer.padding_side == "left"
    with pytest.raises(ValueError, match="vocabulary"):
        policy.install_saved_processor(model, processor({"a": 1, "<|action_0|>": 0}))
    with pytest.raises(ValueError, match="pad_token_id"):
        policy.install_saved_processor(model, processor(pad=1))


def test_unused_decoder_cache_matches_trainer_validation_without_other_config_changes():
    text = SimpleNamespace(use_cache=True, hidden_size=2048)
    config = SimpleNamespace(use_cache=True, text_config=text, attention_dropout=0.)
    model = SimpleNamespace(qwen_vl_interface=SimpleNamespace(model=SimpleNamespace(config=config)))
    policy.disable_unused_qwen_cache(model)
    assert config.use_cache is False and text.use_cache is False
    assert text.hidden_size == 2048 and config.attention_dropout == 0.
    # A flat decoder config is supported too.
    del config.text_config
    config.use_cache = True
    policy.disable_unused_qwen_cache(model)
    assert config.use_cache is False


def test_config_contract_allows_relocation_but_not_changed_prompt_or_dimensions():
    config = dict(seed=42, framework=dict(qwenvl=dict(base_vlm="/old/Qwen3-VL", device_map="cuda", init_from_config=True),
                                        vj2_model=dict(base_encoder="/old/vjepa", init_from_config=True), action_model=dict(action_dim=9)),
                  datasets=dict(vla_data=dict(resolution_size=224, CoT_prompt="exact trained prompt")),
                  trainer=dict(max_train_steps=1000))
    framework = copy.deepcopy(config["framework"])
    framework["qwenvl"].pop("base_vlm")
    framework["qwenvl"].pop("device_map")
    framework["vj2_model"].pop("base_encoder")
    contract = dict(seed=42, framework=framework, data=config["datasets"]["vla_data"].copy(), trainer=config["trainer"].copy())
    policy.validate_config_contract(config, contract)
    config["framework"]["qwenvl"]["base_vlm"] = "/new/Qwen3-VL"
    policy.validate_config_contract(config, contract)
    config["datasets"]["vla_data"]["CoT_prompt"] = "different"
    with pytest.raises(ValueError, match="training contract"):
        policy.validate_config_contract(config, contract)


def test_predict_mock_keeps_camera_order_state_shape_and_raw_output():
    raw9 = pose.pose8_to_pose9(np.array([.1, .2, .3, 1, 0, 0, 0, .009], np.float32))
    predicted = np.repeat(raw9[None, None], 7, axis=1)

    class MockModel:
        def eval(self):
            return self

        def predict_action(self, *, batch_images, instructions, state):
            assert instructions == ["Press 35 floor."]
            assert batch_images[0][0].getpixel((0, 0)) == (255, 0, 0)
            assert batch_images[0][1].getpixel((0, 0)) == (0, 255, 0)
            assert all(image.size == (224, 224) and image.mode == "RGB" for image in batch_images[0])
            np.testing.assert_array_equal(state, raw9.reshape(1, 1, 9))
            assert not torch.is_grad_enabled()
            return {"normalized_actions": predicted}

    loaded = policy.PiperPolicy.__new__(policy.PiperPolicy)
    loaded._weights_loaded = True
    loaded.device = torch.device("cpu")  # CPU permitted only by the mock; constructor requires CUDA.
    loaded.model = MockModel()
    loaded.resolution = 224
    loaded.controller_gripper_width_m = .008
    inputs = dict(global_image=Image.new("RGB", (640, 480), "red"), wrist_image=Image.new("RGB", (640, 480), (0, 255, 0)),
                  state=raw9, task="Press 35 floor.")
    result = loaded.predict(**inputs, seed=42)
    np.testing.assert_array_equal(result["raw_pose9"], predicted[0])
    assert result["pose8"].shape == (7, 8) and result["gripper_is_learned"] is False
    with pytest.raises(ValueError, match="exact trained task"):
        loaded.predict(**dict(inputs, task="Press 36 floor."))
    loaded._weights_loaded = False
    with pytest.raises(RuntimeError, match="trained-weight"):
        loaded.predict(**inputs)


def test_comparison_excludes_padding():
    target = np.tile([.1, .2, .3, 1, 0, 0, 0, 1, 0], (7, 1)).astype(np.float32)
    prediction = target.copy()
    prediction[1:, :3] += 100
    metrics = policy._comparison(prediction, target, [False] + [True] * 6)
    assert metrics == dict(valid_target_steps=1, xyz_mae_m=0., rotation_geodesic_mean_deg=0.)
