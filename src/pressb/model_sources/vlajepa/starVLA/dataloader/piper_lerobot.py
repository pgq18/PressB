"""Native, read-only LeRobot v3 adapter for the PiPER press-only collection.

Dependencies: NumPy, PyArrow, PyAV and Pillow; neither torch nor LeRobot is
imported. Camera order is always global, wrist. Input poses are base_link TCP
xyz (metres), quaternion wxyz, gripper opening. Output poses retain xyz exactly
and replace the quaternion with PyTorch3D's first-two-ROWS rotation 6D; opening
is dropped. No normalization, relative-action conversion or IK is performed.

At anchor i, image/state describe i, video covers i..i+7, and action covers
stored action[i:i+7]. Nonterminal actions target the next observation; the
stored terminal action holds the last executed planned endpoint, with no
further real observation. Each requested index is clamped independently to
its own episode's last row and is padded only when it exceeds that row.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image


CAMERA_KEYS = ("observation.images.global", "observation.images.wrist")
POSE8_NAMES = ("x_m", "y_m", "z_m", "qw", "qx", "qy", "qz", "gripper_width_m")
POSE9_NAMES = ("x_m", "y_m", "z_m", "r00", "r01", "r02", "r10", "r11", "r12")
ROTATION_CONVENTION = "pytorch3d_first_two_rows_row_major"


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def quaternion_wxyz_to_matrix(quaternion):
    """Normalize finite nonzero wxyz quaternions; return (...,3,3) rotations."""
    q = np.asarray(quaternion, dtype=np.float64)
    _require(q.shape[-1:] == (4,) and np.isfinite(q).all(), "Expected finite (...,4) wxyz quaternions")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    _require(np.all(norm > 1e-12), "Zero quaternion is not a rotation")
    w, x, y, z = np.moveaxis(q / norm, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=-1).reshape(q.shape[:-1] + (3, 3))


def matrix_to_rotation_6d(matrix):
    """PyTorch3D convention: R[..., :2, :].reshape(..., 6), not columns."""
    matrix = np.asarray(matrix)
    _require(matrix.shape[-2:] == (3, 3) and np.isfinite(matrix).all(), "Expected finite (...,3,3) matrix")
    return matrix[..., :2, :].reshape(matrix.shape[:-2] + (6,)).copy()


def rotation_6d_to_matrix(rotation):
    """Gram-Schmidt the two rows, returning SO(3); reject degenerate predictions."""
    value = np.asarray(rotation, dtype=np.float64)
    _require(value.shape[-1:] == (6,) and np.isfinite(value).all(), "Expected finite (...,6) rotation")
    a, b = value[..., :3], value[..., 3:]
    norm = np.linalg.norm(a, axis=-1, keepdims=True)
    _require(np.all(norm > 1e-10), "Degenerate rotation 6D first row")
    row0 = a / norm
    row1 = b - np.sum(row0 * b, axis=-1, keepdims=True) * row0
    norm = np.linalg.norm(row1, axis=-1, keepdims=True)
    _require(np.all(norm > 1e-10), "Degenerate rotation 6D parallel rows")
    row1 /= norm
    return np.stack((row0, row1, np.cross(row0, row1)), axis=-2)


def matrix_to_quaternion_wxyz(matrix):
    """Stable SO(3) -> unit wxyz conversion, including rotations near pi."""
    m = np.asarray(matrix, dtype=np.float64)
    _require(m.shape[-2:] == (3, 3) and np.isfinite(m).all(), "Expected finite (...,3,3) rotation")
    _require(np.allclose(m @ np.swapaxes(m, -1, -2), np.eye(3), atol=1e-6, rtol=0)
             and np.allclose(np.linalg.det(m), 1., atol=1e-6, rtol=0), "Matrix is not in SO(3)")
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    magnitudes = np.sqrt(np.maximum(0., np.stack((1+m00+m11+m22, 1+m00-m11-m22,
                                                 1-m00+m11-m22, 1-m00-m11+m22), axis=-1)))
    candidates = np.stack((
        np.stack((magnitudes[..., 0]**2, m[..., 2, 1]-m[..., 1, 2], m[..., 0, 2]-m[..., 2, 0], m[..., 1, 0]-m[..., 0, 1]), axis=-1),
        np.stack((m[..., 2, 1]-m[..., 1, 2], magnitudes[..., 1]**2, m[..., 1, 0]+m[..., 0, 1], m[..., 0, 2]+m[..., 2, 0]), axis=-1),
        np.stack((m[..., 0, 2]-m[..., 2, 0], m[..., 1, 0]+m[..., 0, 1], magnitudes[..., 2]**2, m[..., 2, 1]+m[..., 1, 2]), axis=-1),
        np.stack((m[..., 1, 0]-m[..., 0, 1], m[..., 2, 0]+m[..., 0, 2], m[..., 2, 1]+m[..., 1, 2], magnitudes[..., 3]**2), axis=-1),
    ), axis=-2) / (2 * np.maximum(magnitudes[..., :, None], 1e-12))
    choice = magnitudes.argmax(axis=-1)
    q = np.take_along_axis(candidates, choice[..., None, None], axis=-2)[..., 0, :]
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(q[..., :1] < 0, -q, q)


def rotation_6d_to_quaternion_wxyz(rotation):
    return matrix_to_quaternion_wxyz(rotation_6d_to_matrix(rotation))


def pose8_to_pose9(pose):
    """Keep xyz bit-for-bit at its input floating dtype; drop gripper width."""
    source = np.asarray(pose)
    _require(source.shape[-1:] == (8,) and np.isfinite(source).all(), "Expected finite (...,8) xyz+wxyz+width")
    dtype = np.result_type(source.dtype, np.float32)
    result = np.empty(source.shape[:-1] + (9,), dtype=dtype)
    result[..., :3] = source[..., :3]
    result[..., 3:] = matrix_to_rotation_6d(quaternion_wxyz_to_matrix(source[..., 3:7]))
    return result


def _build_split(entries, export_sha256, seed):
    """Exact 90/10 per floor, independent of manifest row ordering or RNG state."""
    _require(type(seed) is int and seed >= 0, "Split seed must be a nonnegative integer")
    ids = [e["episode_id"] for e in entries]
    _require(len(ids) == len(set(ids)) == 1200, "Expected 1200 distinct source episodes")
    train, val = [], []
    for floor in range(24, 36):
        group = sorted(e["episode_id"] for e in entries if e["floor"] == floor)
        _require(len(group) == 100, f"Expected 100 episodes for floor {floor}")
        # Per-floor seed makes changing loop order harmless.
        shuffled = np.random.default_rng(np.random.SeedSequence([seed, floor])).permutation(group).tolist()
        val.extend(shuffled[:10])
        train.extend(shuffled[10:])
    return dict(schema_version=1, kind="piper_episode_stratified_split", seed=seed,
                source_export_manifest_sha256=export_sha256,
                camera_keys=list(CAMERA_KEYS), pose_names=list(POSE9_NAMES), rotation_convention=ROTATION_CONVENTION,
                normalization="none; xyz remains in base_link metres",
                train_source_episode_ids=sorted(train), val_source_episode_ids=sorted(val),
                floor_counts={str(f): {"train": 90, "val": 10} for f in range(24, 36)})


def _split_manifest(expected, requested, root):
    if requested is None:
        return expected
    if isinstance(requested, dict):
        _require(requested == expected, "Split manifest differs from the dataset, seed or stratified partition")
        return expected
    path = Path(requested).resolve()
    _require(path != root and root not in path.parents, "Save split manifests outside the immutable dataset")
    if path.exists():
        _require(json.loads(path.read_text()) == expected, "Existing split manifest differs; refusing to mix partitions")
        return expected
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(expected, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:  # Several distributed ranks may initialize concurrently.
            _require(json.loads(path.read_text()) == expected, "Concurrent split manifest differs")
    finally:
        temporary.unlink()
    return expected


class PiperLeRobotDataset:
    """Map-style torch-compatible dataset without a torch dependency.

    ``sample_stride`` subsamples anchor rows only (always including the last
    row). It does not change the 30 Hz future-video/action timestep. Call
    ``get_frame(source_episode_id, frame_index)`` for any frame in this split.
    Numeric arrays occupy about 22 MB for this dataset and are loaded once;
    resized RGB and open decoders use bounded, process-local LRU caches.
    """

    camera_keys = CAMERA_KEYS
    state_dim = action_dim = 9

    def __init__(self, root, split="train", split_manifest=None, action_horizon=7,
                 video_horizon=8, resolution=224, seed=42, sample_stride=1,
                 frame_cache_size=256, decoder_cache_size=4):
        self.root = Path(root).resolve()
        _require(split in ("train", "val"), "split must be train or val")
        for name, value in (("action_horizon", action_horizon), ("video_horizon", video_horizon),
                            ("sample_stride", sample_stride), ("decoder_cache_size", decoder_cache_size)):
            _require(type(value) is int and value > 0, f"{name} must be a positive integer")
        _require(type(frame_cache_size) is int and frame_cache_size >= 0, "frame_cache_size must be nonnegative")
        self.action_horizon, self.video_horizon = action_horizon, video_horizon
        self.sample_stride, self.split = sample_stride, split
        self.frame_cache_size, self.decoder_cache_size = frame_cache_size, decoder_cache_size
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        _require(len(resolution) == 2 and all(type(v) is int and v > 0 for v in resolution), "resolution must be positive H,W")
        self.resolution = tuple(resolution)
        self.info = json.loads((self.root / "meta/info.json").read_text())
        manifest_path = self.root / "meta/export_manifest.json"
        self.manifest = json.loads(manifest_path.read_text())
        self.export_manifest_sha256 = _sha256(manifest_path)
        _require(self.info["codebase_version"] == "v3.0" and self.info["fps"] == 30, "Expected LeRobot v3 at 30 Hz")
        self.fps = 30
        _require(self.manifest["kind"] == "press_prefix_aggregate", "Expected the published first-press dataset")
        semantics = self.manifest["semantics"]
        _require(semantics["pose_frame"] == "base_link" and semantics["pose_link"] == "gripper_tcp"
                 and semantics["tcp_offset_link6_m"] == [0, 0, .1358] and semantics["quaternion_order"] == "wxyz"
                 and semantics["terminal_action"] == "repeat_previous_planned_target", "Unexpected source pose/action semantics")
        for key in ("observation.state", "action"):
            feature = self.info["features"][key]
            _require(feature["dtype"] == "float32" and feature["shape"] == [8]
                     and tuple(feature["names"]) == POSE8_NAMES, f"Unexpected source feature {key}")
        for key in CAMERA_KEYS:
            _require(self.info["features"][key]["shape"] == [480, 640, 3], "Unexpected camera resolution")
        entries = self.manifest["episodes"]
        expected = _build_split(entries, self.export_manifest_sha256, seed)
        self.split_manifest = _split_manifest(expected, split_manifest, self.root)
        self.split_manifest_sha256 = _canonical_sha(self.split_manifest)
        self.source_episode_ids = tuple(self.split_manifest[f"{split}_source_episode_ids"])
        selected = set(self.source_episode_ids)
        columns = ["episode_index", "length", "tasks", "dataset_from_index", "dataset_to_index",
                   "data/chunk_index", "data/file_index"]
        for key in CAMERA_KEYS:
            columns.extend(f"videos/{key}/{name}" for name in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"))
        episodes = [row for path in sorted((self.root / "meta/episodes").rglob("*.parquet"))
                    for row in pq.read_table(path, columns=columns).to_pylist()]
        episodes.sort(key=lambda e: e["episode_index"])
        _require(len(episodes) == len(entries) == self.info["total_episodes"], "Episode metadata count mismatch")
        self._episodes, self._all_episodes = {}, []
        self._anchors, self._ends = [], []
        end = 0
        for index, (meta, entry) in enumerate(zip(episodes, entries)):
            n = entry["frames"]
            _require(meta["episode_index"] == index and meta["length"] == n
                     and meta["dataset_from_index"] == end and meta["dataset_to_index"] == end + n
                     and meta["tasks"] == [entry["task"]], "Episode ordering, lengths or tasks differ")
            _require(entry["task"] == f"Press {entry['floor']} floor.", "Unexpected task text")
            episode = {**meta, "source_episode_id": entry["episode_id"], "floor": entry["floor"], "task": entry["task"]}
            episode["videos"] = {}
            for key in CAMERA_KEYS:
                prefix = f"videos/{key}/"
                first = float(meta[prefix + "from_timestamp"]) * self.fps
                last = float(meta[prefix + "to_timestamp"]) * self.fps
                _require(abs(first - round(first)) < 1e-3 and abs(last-first-n) < 1e-3, "Video timestamp interval does not equal episode length")
                path = self.root / self.info["video_path"].format(video_key=key,
                    chunk_index=meta[prefix + "chunk_index"], file_index=meta[prefix + "file_index"])
                _require(path.is_file(), f"Missing video {path}")
                episode["videos"][key] = (str(path), int(round(first)))
            self._all_episodes.append(episode)
            if entry["episode_id"] in selected:
                self._episodes[entry["episode_id"]] = episode
                anchors = list(range(0, n, sample_stride))
                if anchors[-1] != n - 1:
                    anchors.append(n - 1)
                self._anchors.append((entry["episode_id"], np.asarray(anchors, dtype=np.int64)))
                self._ends.append((self._ends[-1] if self._ends else 0) + len(anchors))
            end += n
        _require(end == self.info["total_frames"], "Episode boundaries do not span the numeric dataset")
        self.episode_lengths = {sid: e["length"] for sid, e in self._episodes.items()}
        self._load_numeric(end)
        self._pid = os.getpid()
        self._frames, self._decoders = OrderedDict(), OrderedDict()

    def _load_numeric(self, total):
        self._states = np.empty((total, 9), dtype=np.float32)
        self._actions = np.empty((total, 9), dtype=np.float32)
        expected_episode = np.empty(total, dtype=np.int32)
        expected_frame = np.empty(total, dtype=np.int32)
        expected_source = np.empty(total, dtype=np.int32)
        for e in self._all_episodes:
            a, b = e["dataset_from_index"], e["dataset_to_index"]
            expected_episode[a:b], expected_source[a:b] = e["episode_index"], e["source_episode_id"]
            expected_frame[a:b] = np.arange(e["length"])
        seen = np.zeros(total, dtype=bool)
        columns = ["index", "episode_index", "frame_index", "source_episode_id", "timestamp", "observation.state", "action"]
        for path in sorted((self.root / "data").rglob("*.parquet")):
            table = pq.read_table(path, columns=columns)
            indices = np.asarray(table["index"].to_pylist(), dtype=np.int64).reshape(-1)
            _require(len(indices) and indices.min() >= 0 and indices.max() < total
                     and len(np.unique(indices)) == len(indices) and not seen[indices].any(), "Overlapping or out-of-range numeric rows")
            for key, expected in (("episode_index", expected_episode), ("frame_index", expected_frame), ("source_episode_id", expected_source)):
                actual = np.asarray(table[key].to_pylist()).reshape(-1)
                _require(np.array_equal(actual, expected[indices]), f"Numeric {key} differs from episode metadata")
            _require(np.allclose(np.asarray(table["timestamp"].to_pylist()).reshape(-1), expected_frame[indices]/self.fps,
                                 atol=2e-6, rtol=0), "Numeric timestamps do not match 30 Hz")
            for key, target in (("observation.state", self._states), ("action", self._actions)):
                value = np.asarray(table[key].to_pylist(), dtype=np.float32)
                _require(value.shape == (len(indices), 8) and np.isfinite(value).all()
                         and np.allclose(np.linalg.norm(value[:, 3:7], axis=1), 1., atol=1e-5, rtol=0), "Invalid source TCP poses")
                target[indices] = pose8_to_pose9(value)
            seen[indices] = True
        _require(seen.all(), "Numeric dataset has missing rows")
        for e in self._all_episodes:
            end = e["dataset_to_index"]
            _require(np.array_equal(self._actions[end-1], self._actions[end-2]), "Episode terminal action is not clamped")
        self._states.flags.writeable = self._actions.flags.writeable = False

    def __len__(self):
        return self._ends[-1]

    def sample_metadata(self, index):
        """Resolve a strided sample index without decoding images."""
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self):
            raise IndexError(index)
        part = bisect_right(self._ends, int(index))
        local = int(index) - (self._ends[part - 1] if part else 0)
        source, anchors = self._anchors[part]
        frame = int(anchors[local])
        e = self._episodes[source]
        return dict(source_episode_id=source, episode_index=e["episode_index"], frame_index=frame,
                    dataset_index=e["dataset_from_index"] + frame, floor=e["floor"], timestamp=frame/self.fps)

    def __getitem__(self, index):
        item = self.sample_metadata(index)
        return self.get_frame(item["source_episode_id"], item["frame_index"])

    def _ensure_process(self):
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()

    def _decoder(self, path):
        if path in self._decoders:
            self._decoders.move_to_end(path)
            return self._decoders[path]
        container = av.open(path)
        _require(len(container.streams.video) == 1, f"Expected one RGB video stream: {path}")
        stream = container.streams.video[0]
        _require(abs(float(stream.average_rate) - self.fps) < 1e-6, f"Video fps mismatch: {path}")
        stream.codec_context.thread_count = 1
        self._decoders[path] = (container, stream)
        while len(self._decoders) > self.decoder_cache_size:
            _, (old, _) = self._decoders.popitem(last=False)
            old.close()
        return container, stream

    def _video_frames(self, path, frame_ids):
        self._ensure_process()
        ids = sorted(set(int(i) for i in frame_ids))
        found = {}
        for i in ids:
            key = (path, i)
            if key in self._frames:
                found[i] = self._frames[key]
                self._frames.move_to_end(key)
        missing = set(ids) - set(found)
        if missing:
            container, stream = self._decoder(path)
            first, last = min(missing), max(missing)
            seek_pts = int(np.floor((first / self.fps) / float(stream.time_base)))
            container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
            for frame in container.decode(stream):
                _require(frame.pts is not None and frame.time_base is not None, "Video frame lacks a PTS")
                time = float(frame.pts * frame.time_base)
                index = int(round(time * self.fps))
                if index > last:
                    break
                if index not in missing:
                    continue
                _require(abs(time-index/self.fps) <= .5/self.fps, "Video timestamp error exceeds half a frame")
                rgb = frame.to_ndarray(format="rgb24")
                _require(rgb.shape == (480, 640, 3), "Invalid decoded RGB shape")
                h, w = self.resolution
                rgb = np.asarray(Image.fromarray(rgb).resize((w, h), Image.Resampling.BILINEAR)).copy()
                rgb.flags.writeable = False
                found[index] = rgb
                if self.frame_cache_size:
                    self._frames[(path, index)] = rgb
                    while len(self._frames) > self.frame_cache_size:
                        self._frames.popitem(last=False)
                missing.remove(index)
                if not missing:
                    break
            _require(not missing, f"Video does not contain requested frames {sorted(missing)}: {path}")
        return np.stack([found[int(i)] for i in frame_ids])

    def get_frame(self, source_episode_id, frame_index):
        """Read any frame in this split, including anchors skipped by stride."""
        if source_episode_id not in self._episodes:
            raise KeyError(f"Source episode {source_episode_id} is outside split {self.split}")
        e = self._episodes[source_episode_id]
        if not isinstance(frame_index, (int, np.integer)) or not 0 <= frame_index < e["length"]:
            raise IndexError(frame_index)
        i, final = int(frame_index), e["length"] - 1
        action_requests = i + np.arange(self.action_horizon)
        video_requests = i + np.arange(self.video_horizon)
        a_indices = e["dataset_from_index"] + np.minimum(action_requests, final)
        v_indices = np.minimum(video_requests, final)
        views = []
        for key in CAMERA_KEYS:
            path, offset = e["videos"][key]
            views.append(self._video_frames(path, offset + v_indices))
        video = np.stack(views)
        return dict(image=[Image.fromarray(view[0].copy()) for view in views], video=video,
                    state=self._states[e["dataset_from_index"] + i:i + e["dataset_from_index"] + 1].copy(),
                    action=self._actions[a_indices].copy(), lang=e["task"],
                    action_is_pad=(action_requests > final), video_is_pad=(video_requests > final),
                    source_episode_id=source_episode_id, episode_index=e["episode_index"], frame_index=i,
                    dataset_index=e["dataset_from_index"] + i, floor=e["floor"], timestamp=i/self.fps,
                    video_timestamps=(v_indices / self.fps).astype(np.float64))

    def close(self):
        for container, _ in getattr(self, "_decoders", {}).values():
            container.close()
        self._decoders = OrderedDict()
        self._frames = OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_decoders"], state["_frames"] = OrderedDict(), OrderedDict()
        return state

    def __del__(self):
        self.close()


def collate_fn(batch):
    """starVLA consumes a list of examples; keep PIL images and explicit masks."""
    return batch
