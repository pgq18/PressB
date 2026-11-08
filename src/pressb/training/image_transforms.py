"""Shared, train-only image augmentation for supervised model adapters.

The configuration helpers need no Torch/Isaac/model imports. Construct the
callable before creating DataLoader workers and call it once per camera, with
CHW images or a TCHW clip. A clip receives one set of spatial/color parameters;
derive any current-frame image from that augmented clip. Cache decoded source
images, never augmented ones. Validation/test/inference always return None.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
from numbers import Real


DEFAULT_IMAGE_TRANSFORMS = {
    "enable": True,
    "max_num_transforms": 3,
    "random_order": False,
    "tfs": {
        "brightness": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"brightness": [0.8, 1.2]}},
        "contrast": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"contrast": [0.8, 1.2]}},
        "saturation": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"saturation": [0.8, 1.2]}},
        "hue": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"hue": [-0.05, 0.05]}},
        "sharpness": {"weight": 1.0, "type": "SharpnessJitter", "kwargs": {"sharpness": [0.8, 1.2]}},
        "affine": {"weight": 1.0, "type": "RandomAffine", "kwargs": {"degrees": [-3.0, 3.0], "translate": [0.05, 0.05]}},
    },
}


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain(item) for item in value]
    return deepcopy(value)


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _range(value, name, *, lower=0.0, upper=None, center=1.0):
    """Check torchvision-style scalar jitter or explicit [minimum, maximum]."""
    if isinstance(value, Real) and not isinstance(value, bool):
        delta = _number(value, name)
        if delta < 0 or (upper is not None and delta > upper):
            raise ValueError(f"Invalid {name} range")
        return [max(lower, center - delta), center + delta]
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a number or a two-element range")
    lo, hi = [_number(item, name) for item in value]
    if lo > hi or lo < lower or (upper is not None and hi > upper):
        raise ValueError(f"Invalid {name} range")
    return [lo, hi]


def _validate_kwargs(kind, kwargs):
    allowed = {
        "ColorJitter": {"brightness", "contrast", "saturation", "hue"},
        "SharpnessJitter": {"sharpness"},
        "RandomAffine": {"degrees", "translate"},
    }
    if kind not in allowed:
        raise ValueError(f"Unsupported image transform {kind!r}; extend the shared SFT transform registry")
    if set(kwargs) - allowed[kind]:
        raise ValueError(f"Unsupported {kind} arguments: {sorted(set(kwargs) - allowed[kind])}")
    if kind == "ColorJitter":
        for key, value in kwargs.items():
            if key == "hue":
                _range(value, key, lower=-0.5, upper=0.5, center=0.0)
            else:
                _range(value, key)
    elif kind == "SharpnessJitter":
        if "sharpness" not in kwargs:
            raise ValueError("SharpnessJitter requires sharpness")
        _range(kwargs["sharpness"], "sharpness")
    else:
        if "degrees" not in kwargs:
            raise ValueError("RandomAffine requires degrees")
        _range(kwargs["degrees"], "degrees", lower=-math.inf, center=0.0)
        if "translate" in kwargs:
            value = kwargs["translate"]
            if not isinstance(value, list) or len(value) != 2 or any(
                not 0 <= _number(item, "translate") <= 1 for item in value
            ):
                raise ValueError("translate must contain two fractions in [0, 1]")


def normalize_image_transforms_config(config=None):
    """Resolve the shared preset into a serializable copy for run provenance.

    None or an omitted block selects the preset. Set enable=false to opt out.
    A supplied tfs mapping replaces the candidate set rather than merging it.
    """
    if config is not None and not isinstance(config, Mapping):
        raise ValueError("image_transforms must be a mapping or null")
    override = _plain(config) if config is not None else {}
    if set(override) - set(DEFAULT_IMAGE_TRANSFORMS):
        raise ValueError(f"Unknown image_transforms fields: {sorted(set(override) - set(DEFAULT_IMAGE_TRANSFORMS))}")
    result = {**deepcopy(DEFAULT_IMAGE_TRANSFORMS), **override}
    if type(result["enable"]) is not bool or type(result["random_order"]) is not bool:
        raise ValueError("image_transforms enable and random_order must be booleans")
    if type(result["max_num_transforms"]) is not int or result["max_num_transforms"] < 0:
        raise ValueError("max_num_transforms must be a nonnegative integer")
    if not isinstance(result["tfs"], dict):
        raise ValueError("image_transforms.tfs must be a mapping")
    for name, item in result["tfs"].items():
        if not isinstance(name, str) or not name or not isinstance(item, dict):
            raise ValueError("Each image transform needs a nonempty name and a mapping")
        if set(item) - {"weight", "type", "kwargs"} or "type" not in item:
            raise ValueError(f"Invalid transform configuration for {name}")
        weight = _number(item.get("weight", 1.0), f"{name}.weight")
        if weight < 0:
            raise ValueError(f"{name}.weight must be nonnegative")
        item["weight"] = weight
        item.setdefault("kwargs", {})
        if not isinstance(item["type"], str) or not isinstance(item["kwargs"], dict):
            raise ValueError(f"Invalid type/kwargs for {name}")
        _validate_kwargs(item["type"], item["kwargs"])
    return result


def resolve_image_transforms_config(training_config, *, legacy_config=None):
    """Prefer the common root key; accept an adapter's historical block as fallback."""
    if not isinstance(training_config, Mapping):
        raise ValueError("Training configuration must be a mapping")
    value = training_config["image_transforms"] if "image_transforms" in training_config else legacy_config
    return normalize_image_transforms_config(value)


class _SharpnessJitter:
    def __init__(self, sharpness):
        self.bounds = _range(sharpness, "sharpness")

    def __call__(self, image):
        import torch
        from torchvision.transforms.v2 import functional as F

        factor = float(torch.empty(()).uniform_(*self.bounds))
        return F.adjust_sharpness(image, factor)


class SFTImageTransforms:
    """Weighted random subset, preserving tensor shape/dtype and source storage.

    Direct calls use PyTorch's worker-local RNG. Dataset adapters should use
    apply_to_sample with a stable sample/camera key for reproducible resume.
    Selection is without replacement, with min(max_num_transforms, candidates)
    operations. One TCHW call shares each transform's parameters across frames.
    """
    def __init__(self, config, *, seed=0):
        import torch
        from torchvision.transforms import v2

        self.config = deepcopy(config)
        self.seed = seed
        # Epoch changes reach persistent DataLoader workers without restarting
        # them or putting source RGB/action data into shared memory.
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.transforms = []
        weights = []
        for item in config["tfs"].values():
            if item["weight"] == 0:
                continue
            constructor = _SharpnessJitter if item["type"] == "SharpnessJitter" else getattr(v2, item["type"])
            self.transforms.append(constructor(**item["kwargs"]))
            weights.append(item["weight"])
        # Normalize before constructing float32 weights to avoid overflow.
        scale = max(weights)
        self.weights = torch.tensor([value / scale for value in weights], dtype=torch.float64)
        self.count = min(len(self.transforms), config["max_num_transforms"])

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise ValueError("Augmentation epoch must be a nonnegative integer")
        self._epoch.fill_(epoch)

    def apply_to_sample(self, image, sample_key):
        """Deterministic per epoch/sample, independent of worker count or offset.

        Call set_epoch before creating the epoch's DataLoader iterator. A stable
        key should identify the source episode, anchor frame and camera. Repeated
        reads of the same key in the same epoch intentionally use the same
        augmentation. Forking only the CPU RNG leaves parent/model RNG untouched.
        """
        import torch

        payload = json.dumps([self.seed, int(self._epoch), sample_key],
                             separators=(",", ":"), allow_nan=False).encode()
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            return self(image)

    def __call__(self, image):
        import torch

        if not isinstance(image, torch.Tensor) or image.ndim not in (3, 4) or image.shape[-3] != 3:
            raise ValueError("SFT image transforms expect RGB CHW or TCHW torch tensors")
        if image.dtype not in (torch.uint8, torch.float32, torch.float64):
            raise ValueError("SFT images must be uint8 or floating RGB in [0, 1]")
        indices = torch.multinomial(self.weights, self.count, replacement=False)
        if not self.config["random_order"]:
            indices = indices.sort().values
        output = image.clone()
        for index in indices.tolist():
            output = self.transforms[index](output)
        return output


def build_image_transforms(config=None, *, split="train", seed=0):
    """Build the common train-only pipeline; evaluation consumes no augmentation RNG."""
    if split not in {"train", "training", "val", "validation", "test", "eval", "inference"}:
        raise ValueError(f"Unknown SFT dataset split {split!r}")
    if type(seed) is not int:
        raise ValueError("Augmentation seed must be an integer")
    resolved = normalize_image_transforms_config(config)
    if split not in {"train", "training"} or not resolved["enable"] or not resolved["max_num_transforms"]:
        return None
    if not any(item["weight"] > 0 for item in resolved["tfs"].values()):
        return None
    return SFTImageTransforms(resolved, seed=seed)
