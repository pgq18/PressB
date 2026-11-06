"""Six-action adapter for the 3,000-episode, layout-grouped button datasets.

Source data remains joint7. States retain measured gripper width; only action
supervision excludes the constant gripper command. All three instructions for
an identical colored-button layout remain in one train/validation partition.
"""
from __future__ import annotations

import copy
import numpy as np

from .piper_lerobot import _canonical_sha, _require
from .table_buttons_multitask_lerobot import TableButtonsMultitaskLeRobotDataset, COLORS, CAMERA_KEYS, collate_fn
from .table_buttons_joint6_lerobot import (
    NORMALIZATION, FIXED_GRIPPER_FULL_WIDTH_M, normalize_joint6, denormalize_joint6,
    normalize_joint7,
)


def _build_split(entries, export_sha256, seed):
    _require(type(seed) is int and seed >= 0, "Split seed must be a nonnegative integer")
    _require(len(entries) == 3000 and sorted(e["episode_index"] for e in entries) == list(range(3000)),
             "Expected 3,000 distinct merged episode indices")
    groups, source_ids = {}, {color: set() for color in COLORS}
    for entry in entries:
        color, sid = entry["target_color"], entry["source_episode_index"]
        _require(color in COLORS and type(sid) is int and 0 <= sid < 1000,
                 "Invalid source color or episode index")
        _require(sid not in source_ids[color] and entry["episode_index"] == COLORS.index(color) * 1000 + sid
                 and entry["task_index"] == COLORS.index(color), "Merged episode/task remapping differs")
        source_ids[color].add(sid)
        positions = entry["positions_xy_m"]
        _require(set(positions) == set(COLORS) and all(np.asarray(v).shape == (2,)
                 and np.isfinite(v).all() for v in positions.values()), "Invalid button positions")
        group = _canonical_sha(dict(spacing_profile=entry["spacing_profile"], positions_xy_m=positions))
        _require(entry.get("layout_group_id") == group, "layout_group_id differs from actual colored-button positions")
        _require(not 10000 <= entry["seed"] <= 10099, "Physical evaluation seeds must not occur in training data")
        groups.setdefault(group, []).append(entry)
    _require(len(groups) == 1000, "Expected 1,000 unique physical layouts")
    _require(all(len(rows) == 3 and {e["target_color"] for e in rows} == set(COLORS)
                 for rows in groups.values()), "Each physical layout must have exactly one demonstration per color")
    # Order by the red demonstration's source index, preserving the existing
    # seed42 900/100 split where expanded color collections are perfectly paired.
    ordered = sorted(groups, key=lambda key: next(e["source_episode_index"] for e in groups[key]
                                                 if e["target_color"] == "red"))
    shuffled = np.random.default_rng(seed).permutation(1000).tolist()
    partitions = {"train": shuffled[100:], "val": shuffled[:100]}
    layouts = {split: sorted(ordered[index] for index in indices) for split, indices in partitions.items()}
    episodes = {split: sorted(e["episode_index"] for group in keys for e in groups[group])
                for split, keys in layouts.items()}
    return dict(schema_version=3, kind="table_buttons_multitask_joint6_layout_split", seed=seed,
                source_export_manifest_sha256=export_sha256, camera_keys=list(CAMERA_KEYS),
                normalization=copy.deepcopy(NORMALIZATION),
                train_source_episode_ids=episodes["train"], val_source_episode_ids=episodes["val"],
                layout_group_ids_by_split=layouts, counts={"train": 2700, "val": 300},
                counts_per_color={color: {"train": 900, "val": 100} for color in COLORS},
                layout_group_counts={"train": 900, "val": 100}, layout_group_key="layout_group_id",
                selection="seed42 permutation of 1000 actual layouts ordered by red source episode index; first100 validation",
                excluded_evaluation_seed_range=[10000, 10099])


class TableButtonsMultitaskJoint6LeRobotDataset(TableButtonsMultitaskLeRobotDataset):
    state_dim, action_dim = 7, 6
    expected_episode_count = 3000
    export_schema_version = 3
    build_split = staticmethod(_build_split)
    normalize = staticmethod(normalize_joint6)
    denormalize = staticmethod(denormalize_joint6)
    normalize_state = staticmethod(normalize_joint7)
    normalize_action = staticmethod(normalize_joint6)
    denormalize_action = staticmethod(denormalize_joint6)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        expected = np.float32(FIXED_GRIPPER_FULL_WIDTH_M * 100.) / np.float32(8.)
        _require(np.all(self._actions[:, 6] == expected),
                 "Joint6 policy requires the constant 0.2 mm source gripper command")
        self._actions = self._actions[:, :6].copy()
        self._actions.flags.writeable = False
        self.normalization = copy.deepcopy(NORMALIZATION)
        self.fingerprint = dict(self.fingerprint, adapter=type(self).__name__,
            state_dim=7, action_dim=6, normalization=copy.deepcopy(self.normalization))
        self.fingerprint_sha256 = _canonical_sha(self.fingerprint)
