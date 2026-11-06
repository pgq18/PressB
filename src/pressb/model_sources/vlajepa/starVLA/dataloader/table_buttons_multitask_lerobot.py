"""Read-only LeRobot v3 adapter for the three merged multi-task button datasets.

Input joints are degrees and the seventh coordinate is FULL gripper width in
centimetres. Both measured state and next-interval command are divided by the
fixed scale [180,180,180,180,180,180,8]. There is no centering, clipping, fitted
dataset statistic, relative-action conversion or discarded gripper channel.
The inverse multiplies by the same scale and returns degrees/full-width cm.

Only image decoding and cache lifecycle are inherited from the elevator
adapter. Its constructor, TCP conversion and floor-specific split are never
used. Cameras are ordered up, wrist; all future targets remain at 30 Hz.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
import copy
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from .piper_lerobot import (PiperLeRobotDataset, _require, _sha256,
                            _canonical_sha, _split_manifest, collate_fn)


from .table_buttons_lerobot import (CAMERA_KEYS, JOINT7_NAMES, JOINT7_UNITS,
                                    JOINT7_SCALE, NORMALIZATION, normalize_joint7, denormalize_joint7)


COLORS = ("red", "yellow", "blue")


def _build_split(entries, export_sha256, seed):
    """Keep all three instructions for the same physical layout in one split."""
    _require(type(seed) is int and seed >= 0, "Split seed must be a nonnegative integer")
    _require(len(entries) == 300 and sorted(e["episode_index"] for e in entries) == list(range(300)),
             "Expected 300 distinct merged episode indices 0..299")
    by_color = {color: {} for color in COLORS}
    for entry in entries:
        color, sid = entry["target_color"], entry["source_episode_index"]
        _require(color in COLORS and type(sid) is int and 0 <= sid < 100,
                 "Invalid color or original source episode index")
        _require(sid not in by_color[color] and entry["episode_index"] == COLORS.index(color) * 100 + sid
                 and entry["task_index"] == COLORS.index(color), "Merged episode/task remapping differs")
        by_color[color][sid] = entry
    _require(all(set(rows) == set(range(100)) for rows in by_color.values()), "Missing per-color episodes")
    layout_hashes, seeds = {}, {}
    for sid in range(100):
        reference = by_color[COLORS[0]][sid]
        positions = reference["positions_xy_m"]
        _require(set(positions) == set(COLORS) and all(np.asarray(v).shape == (2,)
                 and np.isfinite(v).all() for v in positions.values()), "Invalid button positions")
        layout = _canonical_sha(positions)
        _require(layout not in layout_hashes and reference["seed"] not in seeds,
                 "Duplicate source layout or seed across original episode indices")
        layout_hashes[layout], seeds[reference["seed"]] = sid, sid
        for color in COLORS[1:]:
            paired = by_color[color][sid]
            _require(paired["seed"] == reference["seed"] and paired["positions_xy_m"] == positions,
                     "Paired instructions must share the same collection seed and layout")
    shuffled = np.random.default_rng(seed).permutation(100).tolist()
    original = {"train": sorted(shuffled[10:]), "val": sorted(shuffled[:10])}
    merged = {split: sorted(i * 100 + sid for i in range(3) for sid in ids)
              for split, ids in original.items()}
    return dict(schema_version=2, kind="table_buttons_multitask_paired_layout_split", seed=seed,
                source_export_manifest_sha256=export_sha256, camera_keys=list(CAMERA_KEYS),
                state_action_names=list(JOINT7_NAMES), normalization=copy.deepcopy(NORMALIZATION),
                train_source_episode_ids=merged["train"], val_source_episode_ids=merged["val"],
                original_episode_ids_by_split=original, counts={"train": 270, "val": 30},
                counts_per_color={color: {"train": 90, "val": 10} for color in COLORS},
                layout_group_key="source_episode_index; exact shared seed and positions verified across colors",
                selection="same original seed42 episode permutation as single-task runs; shared layouts never cross splits")


class TableButtonsMultitaskLeRobotDataset(PiperLeRobotDataset):
    """Map-style dataset returning the existing VLA-JEPA sample dictionary.

    ``sample_stride`` changes anchor selection only. Every video/action chunk
    still uses consecutive 30 Hz frames, clamped within its own episode. Split
    manifests are saved outside the immutable dataset; source statistics are
    neither read nor used for normalization.
    """

    camera_keys = CAMERA_KEYS
    state_dim = action_dim = 7
    expected_episode_count = 300
    export_schema_version = 2
    build_split = staticmethod(_build_split)
    normalize = staticmethod(normalize_joint7)
    denormalize = staticmethod(denormalize_joint7)
    normalize_joint7 = staticmethod(normalize_joint7)
    denormalize_joint7 = staticmethod(denormalize_joint7)

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
        _require(len(resolution) == 2 and all(type(v) is int and v > 0 for v in resolution),
                 "resolution must be positive H,W")
        self.resolution = tuple(resolution)
        info_path = self.root / "meta/info.json"
        manifest_path = self.root / "meta/export_manifest.json"
        self.info = json.loads(info_path.read_text())
        self.manifest = json.loads(manifest_path.read_text())
        self.export_manifest_sha256 = _sha256(manifest_path)
        self.normalization = copy.deepcopy(NORMALIZATION)
        _require(self.info["codebase_version"] == "v3.0" and self.info["fps"] == 30
                 and self.info["total_tasks"] == 3, "Expected three-task LeRobot v3 at 30 Hz")
        self.fps = 30
        m = self.manifest
        _require(m["schema_version"] == self.export_schema_version and m["kind"] == "table_buttons_multitask" and m["format_version"] == "v3.0" and m["fps"] == 30,
                 "Unexpected table-buttons export format")
        self.group, self.spacing_profile = m["group"], m["spacing_profile"]
        self.target_colors, self.tasks = tuple(m["target_colors"]), tuple(m["tasks"])
        _require(self.target_colors == COLORS and self.spacing_profile in ("near", "medium", "far")
                 and self.group == f"{self.spacing_profile}_multitask"
                 and self.tasks == tuple(f"Press the {color} button." for color in COLORS), "Unexpected multi-task identity")
        semantics = m["semantics"]
        _require(semantics["state"] == "measured_joint_position_and_full_gripper_width"
                 and semantics["action"] == "commanded_target_for_following_1_over_30_second_interval"
                 and tuple(semantics["units"]) == JOINT7_UNITS
                 and tuple(semantics["joint_names"]) == JOINT7_NAMES
                 and semantics["terminal_action"] == "repeat_previous_interval_target"
                 and semantics["episode_end"] == "first_30Hz_observation_after_measured_target_press_success",
                 "Unexpected source joint units or action semantics")
        for key in ("observation.state", "action"):
            feature = self.info["features"][key]
            _require(feature["dtype"] == "float32" and feature["shape"] == [7]
                     and tuple(feature["names"]) == JOINT7_NAMES, f"Unexpected source feature {key}")
        for key in CAMERA_KEYS:
            feature = self.info["features"][key]
            _require(feature["dtype"] == "video" and feature["shape"] == [480, 640, 3]
                     and feature["info"]["video.fps"] == 30, "Unexpected camera resolution or fps")
        entries = sorted(m["episodes"], key=lambda e: e["episode_index"])
        expected = self.build_split(entries, self.export_manifest_sha256, seed)
        self.split_manifest = _split_manifest(expected, split_manifest, self.root)
        self.split_manifest_sha256 = _canonical_sha(self.split_manifest)
        self.source_episode_ids = tuple(self.split_manifest[f"{split}_source_episode_ids"])
        selected = set(self.source_episode_ids)
        columns = ["episode_index", "length", "tasks", "dataset_from_index", "dataset_to_index",
                   "data/chunk_index", "data/file_index"]
        for key in CAMERA_KEYS:
            columns.extend(f"videos/{key}/{name}" for name in
                           ("chunk_index", "file_index", "from_timestamp", "to_timestamp"))
        episodes = [row for path in sorted((self.root / "meta/episodes").rglob("*.parquet"))
                    for row in pq.read_table(path, columns=columns).to_pylist()]
        episodes.sort(key=lambda e: e["episode_index"])
        _require(len(episodes) == len(entries) == self.info["total_episodes"] == self.expected_episode_count,
                 "Episode metadata count mismatch")
        self._episodes, self._all_episodes = {}, []
        self._anchors, self._ends = [], []
        end = 0
        for index, (meta, entry) in enumerate(zip(episodes, entries)):
            color, task = entry["target_color"], entry["instruction"]
            _require(task == f"Press the {color} button." and entry["source_group"] == f"{self.spacing_profile}_{color}",
                     "Episode source group or instruction differs")
            n = entry["frame_count"]
            duration, success_time = entry["duration_s"], entry["first_success_time_s"]
            _require(type(n) is int and n >= 2 and np.isfinite([duration, success_time]).all()
                     and abs(duration - (n - 1) / 30) < 1e-6
                     and 0 <= success_time <= duration + 1e-7
                     and duration - success_time < 1 / 30 + 1e-7,
                     "Episode success time or frame count mismatch")
            _require(entry["schema_version"] == "pressb.table_buttons.raw.v1"
                     and entry["success"] is True and entry["stop_reason"] == "success"
                     and entry["terminal_action"] == semantics["terminal_action"]
                     and entry["group"] == self.group and entry["target_color"] in COLORS
                     and entry["spacing_profile"] == self.spacing_profile
                     and entry["instruction"] == task and entry["fps"] == 30
                     and abs(entry["physics_dt"] - 1 / 120) < 1e-12
                     and entry["camera_names"] == ["up", "wrist"]
                     and entry["camera_resolution"] == [640, 480]
                     and entry["state_units"] == ["rad"] * 6 + ["m"]
                     and entry["action_units"] == ["rad"] * 6 + ["m"]
                     and entry["gripper_representation"] == "full_width", "Unexpected raw episode contract")
            _require(meta["episode_index"] == entry["episode_index"] == index and meta["length"] == n
                     and meta["dataset_from_index"] == end and meta["dataset_to_index"] == end + n
                     and meta["tasks"] == [task], "Episode ordering, lengths or tasks differ")
            episode = {**meta, "source_episode_id": index, "group": self.group,
                       "target_color": color, "spacing_profile": self.spacing_profile,
                       "task": task, "task_index": entry["task_index"],
                       "original_source_episode_id": entry["source_episode_index"], "entry": entry, "videos": {}}
            for key in CAMERA_KEYS:
                prefix = f"videos/{key}/"
                first, last = (float(meta[prefix + field]) * self.fps
                               for field in ("from_timestamp", "to_timestamp"))
                _require(np.isfinite([first, last]).all() and first >= 0
                         and abs(first - round(first)) < 1e-3 and abs(last - first - n) < 1e-3,
                         "Video timestamp interval does not equal episode length")
                path = self.root / self.info["video_path"].format(video_key=key,
                    chunk_index=meta[prefix + "chunk_index"], file_index=meta[prefix + "file_index"])
                _require(path.is_file(), f"Missing video {path}")
                episode["videos"][key] = (str(path), int(round(first)))
            self._all_episodes.append(episode)
            if index in selected:
                self._episodes[index] = episode
                anchors = list(range(0, n, sample_stride))
                if anchors[-1] != n - 1:
                    anchors.append(n - 1)
                self._anchors.append((index, np.asarray(anchors, dtype=np.int64)))
                self._ends.append((self._ends[-1] if self._ends else 0) + len(anchors))
            end += n
        _require(end == self.info["total_frames"], "Episode boundaries do not span numeric dataset")
        self.episode_lengths = {sid: e["length"] for sid, e in self._episodes.items()}
        self._load_numeric(end)
        self.fingerprint = dict(adapter="TableButtonsMultitaskLeRobotDataset", version=1,
            source_export_manifest_sha256=self.export_manifest_sha256,
            source_info_sha256=_sha256(info_path), source_numeric_sha256=self._numeric_sha256,
            task=dict(group=self.group, target_colors=list(self.target_colors),
                      spacing_profile=self.spacing_profile, instructions=list(self.tasks)),
            normalization=copy.deepcopy(NORMALIZATION), split_manifest_sha256=self.split_manifest_sha256,
            camera_keys=list(CAMERA_KEYS), fps=self.fps, action_horizon=action_horizon,
            video_horizon=video_horizon, resolution=list(self.resolution), sample_stride=sample_stride)
        self.fingerprint_sha256 = _canonical_sha(self.fingerprint)
        self._pid = os.getpid()
        self._frames, self._decoders = OrderedDict(), OrderedDict()

    def _load_numeric(self, total):
        self._states = np.empty((total, 7), dtype=np.float32)
        self._actions = np.empty((total, 7), dtype=np.float32)
        expected_episode = np.empty(total, dtype=np.int64)
        expected_frame = np.empty(total, dtype=np.int64)
        expected_task = np.empty(total, dtype=np.int64)
        for e in self._all_episodes:
            a, b = e["dataset_from_index"], e["dataset_to_index"]
            expected_episode[a:b], expected_frame[a:b] = e["episode_index"], np.arange(e["length"])
            expected_task[a:b] = e["task_index"]
        seen = np.zeros(total, dtype=bool)
        self._numeric_sha256 = {}
        columns = ["index", "episode_index", "frame_index", "task_index", "timestamp", "observation.state", "action"]
        for path in sorted((self.root / "data").rglob("*.parquet")):
            self._numeric_sha256[str(path.relative_to(self.root))] = _sha256(path)
            table = pq.read_table(path, columns=columns)
            indices = np.asarray(table["index"].to_pylist(), dtype=np.int64).reshape(-1)
            _require(len(indices) and indices.min() >= 0 and indices.max() < total
                     and len(np.unique(indices)) == len(indices) and not seen[indices].any(),
                     "Overlapping or out-of-range numeric rows")
            for key, expected in (("episode_index", expected_episode), ("frame_index", expected_frame)):
                _require(np.array_equal(np.asarray(table[key].to_pylist()).reshape(-1), expected[indices]),
                         f"Numeric {key} differs from episode metadata")
            _require(np.array_equal(np.asarray(table["task_index"].to_pylist()).reshape(-1), expected_task[indices]),
                     "Numeric task index differs from episode instruction")
            _require(np.allclose(np.asarray(table["timestamp"].to_pylist()).reshape(-1),
                                 expected_frame[indices] / self.fps, atol=2e-6, rtol=0),
                     "Numeric timestamps do not match 30 Hz")
            for key, target in (("observation.state", self._states), ("action", self._actions)):
                values = np.asarray(table[key].to_pylist(), dtype=np.float32)
                _require(values.shape == (len(indices), 7) and np.isfinite(values).all(), "Invalid source joint7 values")
                target[indices] = normalize_joint7(values)
            seen[indices] = True
        _require(seen.all(), "Numeric dataset has missing rows")
        for e in self._all_episodes:
            a, b = e["dataset_from_index"], e["dataset_to_index"]
            _require(np.array_equal(self._actions[b - 1], self._actions[b - 2]), "Episode terminal action is not clamped")
            raw_initial = np.asarray(e["entry"]["initial_state_measured_rad_m"], dtype=np.float64)
            _require(raw_initial.shape == (7,) and np.isfinite(raw_initial).all(), "Invalid raw initial state")
            exported_initial = raw_initial * np.asarray([180 / np.pi] * 6 + [100.])
            _require(np.allclose(self._states[a], normalize_joint7(exported_initial), atol=2e-7, rtol=1e-6),
                     "First state does not match measured raw initial joints")
        self._states.flags.writeable = self._actions.flags.writeable = False

    def sample_metadata(self, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self):
            raise IndexError(index)
        part = bisect_right(self._ends, int(index))
        local = int(index) - (self._ends[part - 1] if part else 0)
        source, anchors = self._anchors[part]
        frame = int(anchors[local])
        e = self._episodes[source]
        return dict(source_episode_id=source, episode_index=e["episode_index"], frame_index=frame,
                    dataset_index=e["dataset_from_index"] + frame, group=self.group,
                    target_color=e["target_color"], task_index=e["task_index"],
                    original_source_episode_id=e["original_source_episode_id"], spacing_profile=self.spacing_profile, timestamp=frame / self.fps)

    def get_frame(self, source_episode_id, frame_index):
        if source_episode_id not in self._episodes:
            raise KeyError(f"Source episode {source_episode_id} is outside split {self.split}")
        e = self._episodes[source_episode_id]
        if not isinstance(frame_index, (int, np.integer)) or not 0 <= frame_index < e["length"]:
            raise IndexError(frame_index)
        i, final = int(frame_index), e["length"] - 1
        action_requests, video_requests = i + np.arange(self.action_horizon), i + np.arange(self.video_horizon)
        a_indices = e["dataset_from_index"] + np.minimum(action_requests, final)
        v_indices = np.minimum(video_requests, final)
        views = []
        for key in CAMERA_KEYS:
            path, offset = e["videos"][key]
            views.append(self._video_frames(path, offset + v_indices))
        return dict(image=[Image.fromarray(view[0].copy()) for view in views], video=np.stack(views),
                    state=self._states[e["dataset_from_index"] + i:i + e["dataset_from_index"] + 1].copy(),
                    action=self._actions[a_indices].copy(), lang=e["task"],
                    action_is_pad=action_requests > final, video_is_pad=video_requests > final,
                    source_episode_id=source_episode_id, episode_index=e["episode_index"], frame_index=i,
                    dataset_index=e["dataset_from_index"] + i, group=self.group, target_color=e["target_color"], task_index=e["task_index"],
                    original_source_episode_id=e["original_source_episode_id"],
                    spacing_profile=self.spacing_profile, timestamp=i / self.fps,
                    video_timestamps=(v_indices / self.fps).astype(np.float64))
