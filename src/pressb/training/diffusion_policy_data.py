"""LeRobot v3 episodes and immutable, resized RGB caches for visual SFT.

Units are declared in the training configuration, converted once to SI, and
normalization is fitted only on training episodes. Video offsets are resolved
from each episode's metadata, rather than treating concatenated videos as
individual episodes. Actions are repeated at episode tails, never crossed.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import hashlib
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from pressb.training.image_transforms import build_image_transforms


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LeRobotData:
    def __init__(self, config):
        import pyarrow.parquet as pq

        self.config = dict(config)
        self.root = Path(config["root"]).expanduser().resolve(strict=True)
        self.info = json.loads((self.root / "meta/info.json").read_text())
        if self.info["codebase_version"] != "v3.0":
            raise ValueError("Diffusion Policy currently supports LeRobot v3.0 parquet/video datasets")
        self.fps = float(self.info["fps"])
        self.cameras = list(config.get("cameras", ["up", "wrist"]))
        self.resolution = int(config.get("resolution", 128))
        data_files = sorted((self.root / "data").rglob("*.parquet"))
        meta_files = sorted((self.root / "meta/episodes").rglob("*.parquet"))
        rows = [row for path in data_files for row in pq.read_table(path).to_pylist()]
        rows.sort(key=lambda row: row["index"])
        if [r["index"] for r in rows] != list(range(self.info["total_frames"])):
            raise ValueError("Dataset rows must contain unique consecutive global indices")
        self.episodes = [r for p in meta_files for r in pq.read_table(p).to_pylist()]
        self.episodes.sort(key=lambda row: row["episode_index"])
        if len(self.episodes) != self.info["total_episodes"]:
            raise ValueError("Episode metadata count disagrees with info.json")
        self.task_names = sorted({name for e in self.episodes for name in e["tasks"]})
        self.states = np.asarray([r["observation.state"] for r in rows], dtype=np.float32)
        self.actions = np.asarray([r["action"] for r in rows], dtype=np.float32)
        self.row_episode = np.asarray([r["episode_index"] for r in rows], dtype=np.int64)
        state_scale = np.asarray(config["state_to_si_scale"], dtype=np.float32)
        action_scale = np.asarray(config["action_to_si_scale"], dtype=np.float32)
        if state_scale.shape != (self.states.shape[1],) or action_scale.shape != (self.actions.shape[1],):
            raise ValueError("Explicit dataset-to-SI scales must match feature dimensions")
        self.states *= state_scale
        self.actions *= action_scale
        columns = list(config.get("action_indices", range(self.actions.shape[1])))
        self.actions = self.actions[:, columns]
        if not np.isfinite(self.states).all() or not np.isfinite(self.actions).all():
            raise ValueError("Dataset contains nonfinite states/actions")
        self.task_ids = np.zeros(len(rows), dtype=np.int64)
        self.ends = np.zeros(len(rows), dtype=np.int64)
        expected_begin = 0
        self.video_jobs = defaultdict(dict)
        self.video_counts = defaultdict(int)
        self.timestamp_mode = config.get("video_timestamp_mode", "metadata")
        if self.timestamp_mode not in ("metadata", "packed_validated"):
            raise ValueError("video_timestamp_mode must be metadata or packed_validated")
        for episode in self.episodes:
            begin, end = episode["dataset_from_index"], episode["dataset_to_index"]
            if (begin != expected_begin or end - begin != episode["length"]
                    or len(episode["tasks"]) != 1 or not np.all(self.row_episode[begin:end] == episode["episode_index"])):
                raise ValueError("Inconsistent episode bounds, task or row association")
            expected_begin = end
            self.ends[begin:end] = end - 1
            self.task_ids[begin:end] = self.task_names.index(episode["tasks"][0])
            for camera in self.cameras:
                key = f"observation.images.{camera}"
                prefix = f"videos/{key}/"
                path = self.root / self.info["video_path"].format(video_key=key,
                    chunk_index=episode[prefix + "chunk_index"], file_index=episode[prefix + "file_index"])
                first = float(episode[prefix + "from_timestamp"])
                job = (camera, str(path))
                if self.timestamp_mode == "packed_validated":
                    corrected = self.video_counts[job] / self.fps
                    expected_offset = float(config.get("metadata_timestamp_offset_s", 0))
                    if abs(first - corrected - expected_offset) > 1e-6:
                        raise ValueError("Packed video metadata offset differs from its explicit configured correction")
                    first = corrected
                self.video_counts[job] += end - begin
                times = first + np.arange(end - begin) / self.fps
                indices = np.rint(times * self.fps).astype(np.int64)
                requests = self.video_jobs[(camera, str(path))]
                for video_index, row in zip(indices, range(begin, end)):
                    requests.setdefault(int(video_index), []).append(row)
        if expected_begin != len(rows):
            raise ValueError("Episode ranges do not cover dataset")
        # Include actual video bytes in the identity: same paths/timestamps must
        # not silently reuse an image cache from different recordings.
        files = sorted(set(data_files + meta_files + [self.root / "meta/info.json"]
                           + [Path(p) for _, p in self.video_jobs]))
        self.identity = {str(p.relative_to(self.root)): file_hash(p) for p in files}
        self.sha256 = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()
        self.images = None

    def prepare_images(self, directory):
        import av
        from PIL import Image

        directory = Path(directory).expanduser().resolve()
        if directory.is_relative_to(self.root):
            raise ValueError("Write RGB caches outside the source dataset")
        directory.mkdir(parents=True, exist_ok=True)
        identity = dict(dataset_sha256=self.sha256, resolution=self.resolution,
                        cameras=self.cameras, frames=len(self.states), format="RGB_uint8_nhwc_v1",
                        timestamp_mode=self.timestamp_mode,
                        metadata_timestamp_offset_s=self.config.get("metadata_timestamp_offset_s", 0))
        manifest = directory / "cache.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != identity:
                raise ValueError("Image cache identity does not match dataset or preprocessing")
        else:
            maps = {}
            shape = (len(self.states), self.resolution, self.resolution, 3)
            for camera in self.cameras:
                maps[camera] = np.lib.format.open_memmap(directory / f"{camera}.partial.npy", mode="w+",
                                                        dtype=np.uint8, shape=shape)
            counts = {camera: np.zeros(len(self.states), dtype=np.uint8) for camera in self.cameras}
            for (camera, path), requests in self.video_jobs.items():
                print(json.dumps(dict(event="decode_video", camera=camera, path=path, requested=len(requests))), flush=True)
                with av.open(path) as container:
                    stream = container.streams.video[0]
                    stream.thread_type = "AUTO"
                    stream.codec_context.thread_count = 2
                    decoded_count = 0
                    for frame in container.decode(stream):
                        if frame.time is None:
                            raise ValueError("Video frame has no timestamp")
                        index = int(round(frame.time * self.fps))
                        if self.timestamp_mode == "packed_validated" and abs(frame.time * self.fps - decoded_count) > .01:
                            raise ValueError("Packed correction requires consecutive frames starting at PTS zero")
                        decoded_count += 1
                        if index not in requests:
                            continue
                        rgb = np.asarray(Image.fromarray(frame.to_ndarray(format="rgb24")).resize(
                            (self.resolution, self.resolution), Image.Resampling.BILINEAR))
                        for row in requests[index]:
                            if counts[camera][row]:
                                raise ValueError("Ambiguous duplicate video timestamp")
                            maps[camera][row] = rgb
                            counts[camera][row] = 1
                    if self.timestamp_mode == "packed_validated" and decoded_count != self.video_counts[(camera, path)]:
                        raise ValueError("Packed correction requires video frame count equal to summed episode lengths")
            if any(not count.all() for count in counts.values()):
                missing = {camera: np.flatnonzero(count == 0)[:20].tolist() for camera, count in counts.items()}
                raise ValueError(f"Video frames missing for requested episode timestamps: {missing}")
            for camera, array in maps.items():
                array.flush()
                os.replace(directory / f"{camera}.partial.npy", directory / f"{camera}.npy")
            manifest.write_text(json.dumps(identity, indent=2) + "\n")
        self.images = [np.load(directory / f"{camera}.npy", mmap_mode="r") for camera in self.cameras]
        self.image_paths = [str(directory / f"{camera}.npy") for camera in self.cameras]
        expected = (len(self.states), self.resolution, self.resolution, 3)
        if any(image.shape != expected or image.dtype != np.uint8 for image in self.images):
            raise ValueError("Invalid cached image shape/dtype")

    def __getstate__(self):
        # Spawn workers reopen file-backed arrays rather than serializing the
        # multi-gigabyte RGB cache into each worker process.
        state = self.__dict__.copy()
        state["images"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.__dict__.get("image_paths"):
            self.images = [np.load(path, mmap_mode="r") for path in self.image_paths]

    def split(self, seed=42, validation_fraction=.1):
        ids = [e["episode_index"] for e in self.episodes]
        if not 0 < validation_fraction < 1 or len(ids) < 2:
            raise ValueError("An episode-disjoint validation split requires at least two episodes")
        order = np.random.default_rng(seed).permutation(ids)
        count = max(1, min(len(ids) - 1, round(len(ids) * validation_fraction)))
        return dict(train=sorted(order[count:].tolist()), val=sorted(order[:count].tolist()))

    def normalization(self, train_episode_ids):
        selected = np.isin(self.row_episode, train_episode_ids)
        result = {}
        for name, array in (("state", self.states), ("action", self.actions)):
            lo, hi = array[selected].min(0), array[selected].max(0)
            result[name + "_center"] = ((lo + hi) / 2).tolist()
            result[name + "_scale"] = np.maximum((hi - lo) / 2, 1e-4).tolist()
        return result


class DiffusionDataset(Dataset):
    def __init__(self, data, episode_ids, horizon, *, split="train", image_transforms=None,
                 seed=42, stride=1, max_samples=None):
        self.data, self.horizon = data, int(horizon)
        self.transform = build_image_transforms(image_transforms, split=split, seed=seed)
        anchors = np.flatnonzero(np.isin(data.row_episode, episode_ids))
        self.anchors = anchors[::stride]
        if max_samples is not None and len(self.anchors) > max_samples:
            self.anchors = self.anchors[np.linspace(0, len(self.anchors) - 1, max_samples, dtype=int)]

    def set_epoch(self, epoch):
        if self.transform is not None:
            self.transform.set_epoch(epoch)

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        row = int(self.anchors[index])
        data = self.data
        images = []
        for camera, source in zip(data.cameras, data.images):
            image = torch.from_numpy(source[row].copy()).permute(2, 0, 1)
            if self.transform is not None:
                image = self.transform.apply_to_sample(image, (int(data.row_episode[row]), row, camera))
            images.append(image)
        targets = np.minimum(row + np.arange(self.horizon), data.ends[row])
        return {"images": torch.stack(images), "state": torch.from_numpy(data.states[row].copy()),
                "task": torch.tensor(data.task_ids[row]), "actions": torch.from_numpy(data.actions[targets].copy()),
                "index": torch.tensor(row)}
