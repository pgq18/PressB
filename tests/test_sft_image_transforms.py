"""Shared SFT image behavior, temporal alignment and validation isolation."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from pressb.training.image_transforms import (
    DEFAULT_IMAGE_TRANSFORMS,
    build_image_transforms,
    normalize_image_transforms_config,
    resolve_image_transforms_config,
)


@pytest.fixture
def clip():
    image = torch.randint(25, 220, (3, 24, 32), generator=torch.Generator().manual_seed(5), dtype=torch.uint8)
    return image.unsqueeze(0).repeat(4, 1, 1, 1)


@pytest.mark.parametrize("name", list(DEFAULT_IMAGE_TRANSFORMS["tfs"]))
def test_each_real_transform_preserves_clip_alignment_and_cached_source(name, clip):
    config = {"tfs": {name: DEFAULT_IMAGE_TRANSFORMS["tfs"][name]}, "max_num_transforms": 1}
    transform = build_image_transforms(config, split="train")
    source = clip.clone()
    torch.manual_seed(71)
    results = [transform(clip) for _ in range(4)]
    # A sampled factor near 1 can round to identity on uint8; that is valid.
    assert any(not torch.equal(result, source) for result in results)
    for result in results:
        assert result.shape == source.shape and result.dtype == source.dtype
        assert torch.equal(clip, source)
        assert all(torch.equal(result[0], frame) for frame in result[1:])
        assert result.data_ptr() != clip.data_ptr()


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32, torch.float64])
def test_seed_reproducibility_and_fresh_randomness(dtype, clip):
    image = clip[0] if dtype == torch.uint8 else clip[0].to(dtype) / 255
    transform = build_image_transforms()
    torch.manual_seed(18)
    first = transform(image)
    torch.manual_seed(18)
    repeated = transform(image)
    assert torch.equal(first, repeated)
    assert first.shape == image.shape and first.dtype == image.dtype
    assert any(not torch.equal(first, transform(image)) for _ in range(4))


@pytest.mark.parametrize("split", ["val", "validation", "test", "eval", "inference"])
def test_nontraining_split_never_builds_augmentation_or_consumes_rng(split):
    before = torch.random.get_rng_state().clone()
    assert build_image_transforms({"enable": True}, split=split) is None
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("config", [{"enable": False}, {"max_num_transforms": 0}, {"tfs": {}},
    {"tfs": {"brightness": {"type": "ColorJitter", "weight": 0, "kwargs": {"brightness": [0.5, 0.5]}}}}])
def test_disabled_pipeline_is_rng_neutral(config):
    before = torch.random.get_rng_state().clone()
    assert build_image_transforms(config) is None
    assert torch.equal(before, torch.random.get_rng_state())


def test_explicit_candidates_replace_defaults_and_zero_weights_are_excluded():
    config = {"max_num_transforms": 3, "tfs": {
        "brighten": {"weight": 0, "type": "ColorJitter", "kwargs": {"brightness": [2., 2.]}},
        "darken": {"weight": 1, "type": "ColorJitter", "kwargs": {"brightness": [0.5, 0.5]}},
    }}
    transform = build_image_transforms(config)
    image = torch.full((3, 12, 16), 100, dtype=torch.uint8)
    assert torch.equal(transform(image), torch.full_like(image, 50))


def test_config_is_serializable_independent_and_root_overrides_legacy():
    legacy = {"enable": False}
    assert resolve_image_transforms_config({}, legacy_config=legacy)["enable"] is False
    canonical = resolve_image_transforms_config({"image_transforms": {"enable": True}}, legacy_config=legacy)
    assert canonical["enable"] is True
    assert json.loads(json.dumps(canonical)) == canonical
    canonical["tfs"]["brightness"]["kwargs"]["brightness"][0] = 999
    assert normalize_image_transforms_config()["tfs"]["brightness"]["kwargs"]["brightness"] == [0.8, 1.2]
    assert legacy == {"enable": False}


@pytest.mark.parametrize("config", [False, {"enabled": True}, {"enable": "false"},
    {"max_num_transforms": -1}, {"max_num_transforms": True}, {"random_order": 1},
    {"tfs": []}, {"tfs": {"bad": {"type": "Unknown"}}},
    {"tfs": {"bad": {"type": "ColorJitter", "weight": float("nan")}}},
    {"tfs": {"bad": {"type": "ColorJitter", "weight": -1}}},
    {"tfs": {"bad": {"type": "ColorJitter", "kwargs": {"hue": [-1, 1]}}}},
    {"tfs": {"bad": {"type": "ColorJitter", "kwargs": {"brightness": [1.2, 0.8]}}}},
    {"tfs": {"bad": {"type": "RandomAffine", "kwargs": {"degrees": [-3, 3], "translate": [-1, .1]}}}},
    {"tfs": {"bad": {"type": "SharpnessJitter", "kwargs": {}}}},
])
def test_bad_configuration_rejected_even_for_validation(config):
    with pytest.raises(ValueError):
        build_image_transforms(config, split="val")


def test_unknown_split_rejected():
    with pytest.raises(ValueError, match="split"):
        build_image_transforms(split="trian")


def test_config_normalization_does_not_modify_input():
    config = deepcopy(DEFAULT_IMAGE_TRANSFORMS)
    before = deepcopy(config)
    build_image_transforms(config)
    assert config == before


class KeyedImages(torch.utils.data.Dataset):
    """Tiny source for comparing normal and offset/resumed worker reads."""
    def __init__(self, transform):
        self.transform = transform
        self.image = (torch.arange(3 * 16 * 20).reshape(3, 16, 20) % 151 + 30).to(torch.uint8)

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return self.transform.apply_to_sample(self.image, (0, index, "wrist"))


def test_keyed_augmentation_is_rng_neutral_and_changes_with_epoch_or_camera(clip):
    transform = build_image_transforms(seed=14)
    transform.set_epoch(3)
    before = torch.random.get_rng_state().clone()
    expected = transform.apply_to_sample(clip, (42, 5, "wrist"))
    assert torch.equal(before, torch.random.get_rng_state())
    torch.manual_seed(1234)
    assert torch.equal(expected, transform.apply_to_sample(clip, (42, 5, "wrist")))
    assert not torch.equal(expected, transform.apply_to_sample(clip, (42, 5, "global")))
    transform.set_epoch(4)
    assert not torch.equal(expected, transform.apply_to_sample(clip, (42, 5, "wrist")))


def test_worker_restart_offset_and_persistent_epoch_produce_same_images():
    transform = build_image_transforms(seed=131)
    dataset = KeyedImages(transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=1, persistent_workers=True)
    try:
        transform.set_epoch(7)
        first = torch.cat(list(loader))
        expected = torch.stack([dataset[i] for i in range(len(dataset))])
        assert torch.equal(first, expected)
        restored = build_image_transforms(seed=131)
        restored.set_epoch(7)
        offset = torch.utils.data.Subset(KeyedImages(restored), range(4, 8))
        resumed = torch.utils.data.DataLoader(offset, batch_size=2, num_workers=1)
        assert torch.equal(torch.cat(list(resumed)), first[4:])
        transform.set_epoch(8)
        second = torch.cat(list(loader))
        assert not torch.equal(second, first)
        assert torch.equal(second, torch.stack([dataset[i] for i in range(len(dataset))]))
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32, torch.float64])
@pytest.mark.parametrize("is_clip", [False, True])
@pytest.mark.parametrize("factor", [.6, 1.4])
def test_arm_sharpness_keeps_torchvision_semantics(dtype, is_clip, factor, clip, monkeypatch):
    import pressb.training.image_transforms as transforms
    from torchvision.transforms.v2 import functional as F

    image = clip if is_clip else clip[0]
    if dtype != torch.uint8:
        image = image.to(dtype) / 255
    source = image.clone()
    expected = F.adjust_sharpness(image, factor)
    monkeypatch.setattr(transforms.platform, "machine", lambda: "aarch64")
    actual = transforms._SharpnessJitter([factor, factor])(image)
    assert actual.dtype == dtype and actual.shape == image.shape
    # Native and oneDNN convolution accumulation can differ at a uint8 rounding
    # boundary, but the same torchvision operation and factor remain in use.
    torch.testing.assert_close(actual, expected, atol=1 if dtype == torch.uint8 else 2e-6, rtol=0)
    assert not torch.equal(actual, image)
    assert torch.equal(image, source)
    if is_clip:
        assert all(torch.equal(actual[0], frame) for frame in actual[1:])


@pytest.mark.parametrize("architecture,device,expected_enabled", [
    ("aarch64", "cpu", False), ("arm64", "cpu", False),
    ("x86_64", "cpu", True), ("aarch64", "cuda", True),
    ("arm64", "npu", True),
])
def test_sharpness_backend_workaround_is_only_arm_cpu(architecture, device, expected_enabled, monkeypatch):
    import pressb.training.image_transforms as transforms
    from torchvision.transforms.v2 import functional as F

    monkeypatch.setattr(transforms.platform, "machine", lambda: architecture)
    image = SimpleNamespace(device=SimpleNamespace(type=device))
    observed = []

    def adjust_sharpness(value, factor):
        observed.append(torch.backends.mkldnn.enabled)
        return value

    monkeypatch.setattr(F, "adjust_sharpness", adjust_sharpness)
    with torch.backends.mkldnn.flags(enabled=True):
        assert transforms._SharpnessJitter([.8, .8])(image) is image
        assert torch.backends.mkldnn.enabled is True
    assert observed == [expected_enabled]


def test_arm_sharpness_avoids_jit_failure_and_restores_backend_on_error(monkeypatch):
    import pressb.training.image_transforms as transforms
    from torchvision.transforms.v2 import functional as F

    monkeypatch.setattr(transforms.platform, "machine", lambda: "aarch64")
    original = F.adjust_sharpness

    def affected_grouped_convolution(image, factor):
        if torch.backends.mkldnn.enabled:
            raise RuntimeError("label is too far (Xbyak::Error)")
        return original(image, factor)

    monkeypatch.setattr(F, "adjust_sharpness", affected_grouped_convolution)
    image = torch.arange(3 * 12 * 16).reshape(3, 12, 16).remainder(251).to(torch.uint8)
    transform = transforms._SharpnessJitter([1.4, 1.4])
    with torch.backends.mkldnn.flags(enabled=True, deterministic=True):
        assert transform(image).shape == image.shape
        assert torch.backends.mkldnn.enabled and torch.backends.mkldnn.deterministic

        def another_error(*args):
            assert not torch.backends.mkldnn.enabled
            raise ValueError("invalid image")

        monkeypatch.setattr(F, "adjust_sharpness", another_error)
        with pytest.raises(ValueError, match="invalid image"):
            transform(image)
        assert torch.backends.mkldnn.enabled and torch.backends.mkldnn.deterministic
