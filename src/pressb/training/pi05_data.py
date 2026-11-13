"""PressB supervision data for LeRobot π0.5."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from pressb.training.image_transforms import build_image_transforms


STATE = "observation.state"
ACTION = "action"


def episode_split(total, episodes, fraction, seed):
    ids = list(range(total)) if episodes is None else list(episodes)
    if len(ids) < 2 or len(ids) != len(set(ids)) or any(type(i) is not int or not 0 <= i < total for i in ids):
        raise ValueError("π0.5 SFT needs at least two distinct valid episode indices")
    if not 0 < fraction < 1:
        raise ValueError("dataset.validation_fraction must be in (0, 1)")
    order = np.random.default_rng(seed).permutation(sorted(ids))
    count = max(1, min(len(ids) - 1, round(len(ids) * fraction)))
    return {"train": sorted(order[count:].tolist()), "val": sorted(order[:count].tolist())}


def wire_metadata(dataset, config):
    """Describe how raw LeRobot coordinates map to the PressB transport."""
    if not isinstance(config, Mapping):
        raise ValueError("π0.5 training requires the pressb observation/action contract")
    observation, action = dict(config["observation"]), dict(config["action"])
    camera_keys = dict(config["camera_keys"])
    state = observation["state"]
    if (not camera_keys or observation["cameras"] != list(camera_keys)
            or any(not isinstance(key, str) or not key.startswith("observation.images.") for key in camera_keys.values())
            or len(set(camera_keys.values())) != len(camera_keys)
            or not set(camera_keys.values()).issubset(dataset.camera_keys)):
        raise ValueError("pressb.camera_keys must map the ordered observation cameras to distinct LeRobot RGB keys")
    for label, field, wire in (("state", STATE, state), ("action", ACTION, action)):
        names, units = wire["names"], wire["units"]
        if (not isinstance(names, list) or not isinstance(units, list) or not names or len(units) != len(names)
                or any(not isinstance(x, str) or not x for x in (*names, *units))
                or len(names) != len(set(names))
                or list(dataset.features[field]["shape"]) != [len(names)]):
            raise ValueError(f"pressb {label} names/units must match the LeRobot feature dimension")
    if not np.isclose(float(action["control_hz"]), dataset.fps):
        raise ValueError("pressb.action.control_hz must match the dataset FPS")
    action["control_hz"] = float(dataset.fps)
    for key in ("representation", "frame"):
        if not isinstance(action[key], str) or not action[key]:
            raise ValueError(f"pressb.action.{key} must be a nonempty string")
    result = dict(observation=observation, camera_keys=camera_keys, action=action)
    for key, dim in (("state_scale", len(state["names"])), ("action_scale", len(action["names"]))):
        scale = np.asarray(config.get(key, np.ones(dim)), dtype=float)
        if scale.shape != (dim,) or not np.isfinite(scale).all() or np.any(scale == 0):
            raise ValueError(f"pressb.{key} must be a finite nonzero factor for each coordinate")
        result[key] = scale.tolist()
    return result


def make_datasets(config, policy, *, seed, transforms, wire_config):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    root = config.get("root")
    root = str(Path(root).expanduser().resolve(strict=True)) if root else None
    repo_id = config.get("repo_id") or (Path(root).name if root else None)
    if not repo_id:
        raise ValueError("dataset.root or dataset.repo_id is required")
    options = dict(repo_id=repo_id, root=root, revision=config.get("revision"))
    metadata = LeRobotDatasetMetadata(**options)
    split = episode_split(metadata.total_episodes, config.get("episodes"),
                          float(config.get("validation_fraction", .1)), seed)
    wire = wire_metadata(metadata, wire_config)
    options.update(delta_timestamps={ACTION: [i / metadata.fps for i in policy.action_delta_indices]},
                   video_backend=config.get("video_backend"), tolerance_s=float(config.get("tolerance_s", 1e-4)))
    train = LeRobotDataset(**options, episodes=split["train"])
    val = LeRobotDataset(**options, episodes=split["val"])
    camera_keys = list(wire["camera_keys"].values())
    return (
        PI05Dataset(train, camera_keys, transforms, seed=seed, split="train"),
        PI05Dataset(val, camera_keys, transforms, seed=seed, split="val"),
        split,
        wire,
    )


class PI05Dataset(Dataset):
    def __init__(self, source, camera_keys, transforms, *, seed, split):
        self.source = source
        self.camera_keys = camera_keys
        self.transform = build_image_transforms(transforms, seed=seed, split=split)

    def __len__(self):
        return len(self.source)

    def set_epoch(self, epoch):
        if self.transform is not None:
            self.transform.set_epoch(epoch)

    def __getitem__(self, index):
        item = self.source[index]
        result = {key: item[key] for key in (STATE, ACTION, "task")}
        for key in self.camera_keys:
            image = item[key]
            if self.transform is not None:
                image = self.transform.apply_to_sample(image, (int(item["episode_index"]), int(item["index"]), key))
            result[key] = image
        return result


def fit_stats(source):
    """Fit all native numeric normalization modes from training rows only."""
    numeric = source.hf_dataset.select_columns([STATE, ACTION]).with_format("numpy")
    stats = {}
    for key in (STATE, ACTION):
        rows = np.asarray(numeric[key][:], dtype=np.float32)
        if rows.ndim != 2 or len(rows) == 0 or not np.isfinite(rows).all():
            raise ValueError(f"Nonfinite or malformed training feature {key}")
        quantiles = np.quantile(rows, [.01, .10, .90, .99], axis=0)
        fields = dict(mean=rows.mean(0), std=rows.std(0), min=rows.min(0), max=rows.max(0))
        fields.update(zip(("q01", "q10", "q90", "q99"), quantiles, strict=True))
        stats[key] = {name: torch.as_tensor(values, dtype=torch.float32) for name, values in fields.items()}
    return stats
