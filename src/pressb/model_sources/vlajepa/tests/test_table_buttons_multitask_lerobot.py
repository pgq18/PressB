"""CPU contract tests, with real multi-task-dataset reads when the data is present."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

import av
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


# Load only the native readers, avoiding starVLA's model/data package imports.
MODULE_DIR = Path(__file__).resolve().parents[1] / "starVLA/dataloader"
PACKAGE = "_table_buttons_multitask_reader_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(MODULE_DIR)]
sys.modules[PACKAGE] = package
SPEC = importlib.util.spec_from_file_location(f"{PACKAGE}.table_buttons_multitask_lerobot", MODULE_DIR / "table_buttons_multitask_lerobot.py")
buttons = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = buttons
SPEC.loader.exec_module(buttons)
DATA_ROOT = Path(os.environ.get("TABLE_BUTTONS_TEST_ROOT", "/home/pengguanqi/Datasets/piper_table_buttons_multitask_20261023"))
GROUPS = [f"{spacing}_multitask" for spacing in ("near", "medium", "far")]
COLORS = ("red", "yellow", "blue")


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def synthetic_root(tmp_path):
    root = tmp_path / "near_multitask"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    info = dict(codebase_version="v3.0", fps=30, total_tasks=3, total_episodes=300,
                data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4", features={})
    for key in ("observation.state", "action"):
        info["features"][key] = dict(dtype="float32", shape=[7], names=list(buttons.JOINT7_NAMES))
    for key in buttons.CAMERA_KEYS:
        info["features"][key] = dict(dtype="video", shape=[480, 640, 3], info={"video.fps": 30})
        video = root / info["video_path"].format(video_key=key, chunk_index=0, file_index=0)
        video.parent.mkdir(parents=True)
        video.touch()  # Synthetic tests intercept decoding; real tests decode actual MP4s.
    manifest = dict(schema_version=2, kind="table_buttons_multitask", format_version="v3.0", fps=30, group="near_multitask",
                    spacing_profile="near", target_colors=list(COLORS), tasks=[f"Press the {c} button." for c in COLORS],
                    semantics=dict(state="measured_joint_position_and_full_gripper_width",
                        action="commanded_target_for_following_1_over_30_second_interval",
                        units=list(buttons.JOINT7_UNITS), joint_names=list(buttons.JOINT7_NAMES),
                        terminal_action="repeat_previous_interval_target",
                        episode_end="first_30Hz_observation_after_measured_target_press_success"), episodes=[])
    episode_rows, data_rows = [], []
    cursor = 0
    for sid in range(300):
        color, original = COLORS[sid // 100], sid % 100
        task = f"Press the {color} button."
        n = 3 + sid % 3
        initial = np.array([sid, 1, 2, 3, 4, 5, .02], dtype=np.float32)
        raw_initial = initial.astype(np.float64) * np.array([np.pi / 180] * 6 + [.01])
        manifest["episodes"].append(dict(schema_version="pressb.table_buttons.raw.v1", episode_index=sid,
            frame_count=n, duration_s=(n - 1) / 30, first_success_time_s=(n - 1) / 30 - 1 / 120,
            success=True, stop_reason="success", terminal_action="repeat_previous_interval_target",
            group="near_multitask", spacing_profile="near", target_color=color, instruction=task,
            source_group=f"near_{color}", source_episode_index=original, task_index=sid // 100,
            positions_xy_m={c: [original / 1000, i / 10] for i, c in enumerate(COLORS)},
            fps=30, physics_dt=1 / 120, camera_names=["up", "wrist"], camera_resolution=[640, 480],
            state_units=["rad"] * 6 + ["m"], action_units=["rad"] * 6 + ["m"],
            gripper_representation="full_width", seed=original, initial_state_measured_rad_m=raw_initial.tolist()))
        if original == 0:
            video_start = cursor
            for key in buttons.CAMERA_KEYS:
                video = root / info["video_path"].format(video_key=key, chunk_index=0, file_index=sid // 100)
                video.touch()
        ep = dict(episode_index=sid, length=n, tasks=[task],
                  dataset_from_index=cursor, dataset_to_index=cursor + n,
                  **{"data/chunk_index": 0, "data/file_index": 0})
        for key in buttons.CAMERA_KEYS:
            ep.update({f"videos/{key}/chunk_index": 0, f"videos/{key}/file_index": sid // 100,
                       f"videos/{key}/from_timestamp": (cursor - video_start) / 30,
                       f"videos/{key}/to_timestamp": (cursor + n - video_start) / 30})
        episode_rows.append(ep)
        for frame in range(n):
            state = initial.copy()
            state[1] += frame
            action = initial.copy()
            action[1] += min(frame + 1, n - 1)
            data_rows.append(dict(index=cursor + frame, episode_index=sid, frame_index=frame,
                task_index=sid // 100, timestamp=np.float32(frame / 30).item(),
                **{"observation.state": state.tolist(), "action": action.tolist()}))
        cursor += n
    info["total_frames"] = cursor
    write_json(root / "meta/info.json", info)
    write_json(root / "meta/export_manifest.json", manifest)
    # Deliberately unusable statistics: the adapter must not touch them.
    (root / "meta/stats.json").write_text("these are not the normalization parameters")
    pq.write_table(pa.Table.from_pylist(episode_rows), root / "meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pylist(data_rows), root / "data/chunk-000/file-000.parquet")
    return root


def fake_decode(path, ids):
    # Distinct camera and absolute video frame IDs expose ordering/offset bugs.
    view = 10 if ".up/" in path else 200
    return np.stack([np.full((2, 3, 3), [view, i % 256, i // 256], dtype=np.uint8) for i in ids])


def test_multitask_split_language_numeric_padding_and_video_file_routes(synthetic_root, tmp_path):
    train = buttons.TableButtonsMultitaskLeRobotDataset(synthetic_root, split_manifest=tmp_path / "run/split.json")
    val = buttons.TableButtonsMultitaskLeRobotDataset(synthetic_root, split="val", split_manifest=train.split_manifest)
    train._video_frames = val._video_frames = fake_decode
    expected_val = sorted(np.random.default_rng(42).permutation(100)[:10].tolist())
    assert len(train.source_episode_ids) == 270 and len(val.source_episode_ids) == 30
    assert not set(train.source_episode_ids) & set(val.source_episode_ids)
    for color_index, color in enumerate(COLORS):
        val_ids = sorted(sid % 100 for sid in val.source_episode_ids if sid // 100 == color_index)
        assert val_ids == expected_val
        for original in (0, 50, 99):
            sid = color_index * 100 + original
            dataset = train if sid in train.source_episode_ids else val
            e = dataset._episodes[sid]
            for frame in (0, e["length"] - 1):
                sample = dataset.get_frame(sid, frame)
                assert sample["lang"] == f"Press the {color} button."
                assert sample["target_color"] == color and sample["task_index"] == color_index
                assert sample["original_source_episode_id"] == original
                assert sample["state"].shape == (1, 7) and sample["action"].shape == (7, 7)
                requests = frame + np.arange(7)
                np.testing.assert_array_equal(sample["action_is_pad"], requests >= e["length"])
                np.testing.assert_array_equal(sample["action"], dataset._actions[e["dataset_from_index"] + np.minimum(requests, e["length"] - 1)])
                for view, key in enumerate(buttons.CAMERA_KEYS):
                    path, offset = e["videos"][key]
                    assert path.endswith(f"file-{color_index:03d}.mp4")
                    np.testing.assert_array_equal(sample["video"][view], fake_decode(path, offset + np.minimum(frame + np.arange(8), e["length"] - 1)))
    assert set(train.fingerprint["task"]["instructions"]) == {f"Press the {c} button." for c in COLORS}
    assert train.normalization["scale"] == [180.] * 6 + [8.]
    assert train[0]["lang"] == "Press the red button."
    with pytest.raises(ValueError, match="outside"):
        buttons.TableButtonsMultitaskLeRobotDataset(synthetic_root, split_manifest=synthetic_root / "split.json")


@pytest.mark.parametrize("corrupt", ["paired_seed", "paired_layout", "duplicate_layout", "duplicate_seed", "task_index", "source_episode", "source_group", "instruction", "missing_episode"])
def test_mixed_task_or_leaking_layout_metadata_rejected(synthetic_root, corrupt):
    path = synthetic_root / "meta/export_manifest.json"
    manifest = json.loads(path.read_text())
    entries = manifest["episodes"]
    if corrupt == "paired_seed": entries[100]["seed"] = 900
    if corrupt == "paired_layout": entries[100]["positions_xy_m"]["red"][0] += .1
    if corrupt == "duplicate_layout":
        for sid in (1, 101, 201): entries[sid]["positions_xy_m"] = copy.deepcopy(entries[0]["positions_xy_m"])
    if corrupt == "duplicate_seed":
        for sid in (1, 101, 201): entries[sid]["seed"] = 0
    if corrupt == "task_index": entries[100]["task_index"] = 0
    if corrupt == "source_episode": entries[100]["source_episode_index"] = 1
    if corrupt == "source_group": entries[100]["source_group"] = "near_red"
    if corrupt == "instruction": entries[100]["instruction"] = "Press the red button."
    if corrupt == "missing_episode": entries.pop()
    write_json(path, manifest)
    with pytest.raises(ValueError):
        buttons.TableButtonsMultitaskLeRobotDataset(synthetic_root)


@pytest.mark.parametrize("corrupt", ["task_index", "episode_index", "terminal", "nonfinite"])
def test_corrupt_multitask_numeric_rows_rejected(synthetic_root, corrupt):
    path = synthetic_root / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    row = next(r for r in rows if r["episode_index"] == 100 and r["frame_index"] == 0)
    if corrupt == "task_index": row["task_index"] = 0
    if corrupt == "episode_index": row["episode_index"] = 0
    if corrupt == "nonfinite": row["action"][0] = float("nan")
    if corrupt == "terminal":
        rows[2]["action"][0] += 1
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError):
        buttons.TableButtonsMultitaskLeRobotDataset(synthetic_root)


@pytest.mark.parametrize("group", GROUPS)
def test_real_merged_all_colors_first_middle_terminal(group):
    root = DATA_ROOT / group
    if not (root / "meta/export_manifest.json").exists():
        pytest.skip("Merged dataset is not available yet")
    before = (root / "meta/export_manifest.json").read_bytes()
    train = buttons.TableButtonsMultitaskLeRobotDataset(root, frame_cache_size=16, decoder_cache_size=2)
    val = buttons.TableButtonsMultitaskLeRobotDataset(root, split="val", split_manifest=train.split_manifest,
                                                    frame_cache_size=16, decoder_cache_size=2)
    try:
        assert len(train.source_episode_ids) == 270 and len(val.source_episode_ids) == 30
        raw = [row for path in sorted((root / "data").rglob("*.parquet")) for row in pq.read_table(path).to_pylist()]
        raw.sort(key=lambda row: row["index"])
        for color_index, color in enumerate(COLORS):
            for original in (0, 50, 99):
                sid = color_index * 100 + original
                dataset = train if sid in train.source_episode_ids else val
                episode = dataset._episodes[sid]
                for frame in (0, episode["length"] // 2, episode["length"] - 1):
                    sample = dataset.get_frame(sid, frame)
                    assert sample["lang"] == f"Press the {color} button."
                    assert sample["video"].shape == (2, 8, 224, 224, 3)
                    index = episode["dataset_from_index"] + frame
                    np.testing.assert_array_equal(sample["state"][0], buttons.normalize_joint7(np.asarray(raw[index]["observation.state"], np.float32)))
                    for future in range(7):
                        row = raw[episode["dataset_from_index"] + min(frame + future, episode["length"] - 1)]
                        assert row["task_index"] == color_index
                        np.testing.assert_array_equal(sample["action"][future], buttons.normalize_joint7(np.asarray(row["action"], np.float32)))
                    for view, key in enumerate(buttons.CAMERA_KEYS):
                        np.testing.assert_array_equal(np.asarray(sample["image"][view]), sample["video"][view, 0])
                        if original != 0:
                            continue
                        path, offset = episode["videos"][key]
                        # Independent linear decoder checks merged source-file selection and offset.
                        container = av.open(path)
                        decoded = next(f for i, f in enumerate(container.decode(video=0)) if i == offset + frame)
                        expected = np.asarray(decoded.to_image().resize((224, 224), Image.Resampling.BILINEAR))
                        container.close()
                        np.testing.assert_array_equal(sample["video"][view, 0], expected)
        assert root.joinpath("meta/export_manifest.json").read_bytes() == before
    finally:
        train.close()
        val.close()
