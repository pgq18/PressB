"""CPU contract tests, with real nine-dataset reads when the data is present."""
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
PACKAGE = "_table_buttons_reader_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(MODULE_DIR)]
sys.modules[PACKAGE] = package
SPEC = importlib.util.spec_from_file_location(f"{PACKAGE}.table_buttons_lerobot", MODULE_DIR / "table_buttons_lerobot.py")
buttons = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = buttons
SPEC.loader.exec_module(buttons)
DATA_ROOT = Path(os.environ.get("TABLE_BUTTONS_TEST_ROOT", "/home/pengguanqi/Datasets/piper_table_buttons_20261023"))
GROUPS = [f"{spacing}_{color}" for spacing in ("near", "medium", "far") for color in ("red", "yellow", "blue")]


def test_fixed_transform_roundtrip_and_constant_gripper():
    values = np.array([[180, -90, 0, 45, -180, 360, .02], [0, 90, 30, -45, 10, 0, .02]], dtype=np.float32)
    normalized = buttons.normalize_joint7(values)
    np.testing.assert_allclose(normalized[0], [1, -.5, 0, .25, -1, 2, .0025], rtol=1e-7)
    np.testing.assert_allclose(buttons.denormalize_joint7(normalized), values, rtol=1e-7)
    assert np.isfinite(normalized).all() and normalized.dtype == np.float32
    assert np.all(normalized[:, -1] == .0025)  # Constant width remains meaningful and finite.
    assert normalized[0, 5] == 2  # No clipping to an empirically chosen range.
    np.testing.assert_array_equal(buttons.normalize_joint7(np.zeros(7)), np.zeros(7))
    precise = np.array([.123456789123] * 7, dtype=np.float64)
    np.testing.assert_allclose(buttons.denormalize_joint7(buttons.normalize_joint7(precise)), precise, rtol=2e-16)


@pytest.mark.parametrize("bad", [np.zeros(6), np.zeros((2, 8)), [0, 0, 0, 0, 0, 0, np.nan], [0, 0, 0, 0, 0, 0, np.inf]])
def test_transform_rejects_bad_shape_or_nonfinite(bad):
    with pytest.raises(ValueError):
        buttons.normalize_joint7(bad)
    with pytest.raises(ValueError):
        buttons.denormalize_joint7(bad)


def test_same_episode_partition_for_all_tasks():
    entries = [dict(episode_index=i) for i in range(100)]
    expected_val = sorted(np.random.default_rng(42).permutation(100)[:10].tolist())
    partitions = []
    for group in GROUPS:
        split = buttons._build_split(entries[::-1], group, 42)
        train, val = set(split["train_source_episode_ids"]), set(split["val_source_episode_ids"])
        assert len(train) == 90 and len(val) == 10 and not train & val
        assert train | val == set(range(100)) and sorted(val) == expected_val
        partitions.append((train, val))
    assert all(part == partitions[0] for part in partitions)
    with pytest.raises(ValueError):
        buttons._build_split(entries[:-1], "hash", 42)
    with pytest.raises(ValueError):
        buttons._build_split(entries, "hash", -1)


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def synthetic_root(tmp_path, request):
    count = getattr(request, "param", 100)
    root = tmp_path / "near_red"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    info = dict(codebase_version="v3.0", fps=30, total_tasks=1, total_episodes=count,
                data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4", features={})
    for key in ("observation.state", "action"):
        info["features"][key] = dict(dtype="float32", shape=[7], names=list(buttons.JOINT7_NAMES))
    for key in buttons.CAMERA_KEYS:
        info["features"][key] = dict(dtype="video", shape=[480, 640, 3], info={"video.fps": 30})
        video = root / info["video_path"].format(video_key=key, chunk_index=0, file_index=0)
        video.parent.mkdir(parents=True)
        video.touch()  # Synthetic tests intercept decoding; real tests decode actual MP4s.
    manifest = dict(schema_version=1, format_version="v3.0", fps=30, group="near_red",
                    spacing_profile="near", target_color="red", instruction="Press the red button.",
                    semantics=dict(state="measured_joint_position_and_full_gripper_width",
                        action="commanded_target_for_following_1_over_30_second_interval",
                        units=list(buttons.JOINT7_UNITS), joint_names=list(buttons.JOINT7_NAMES),
                        terminal_action="repeat_previous_interval_target",
                        episode_end="first_30Hz_observation_after_measured_target_press_success"), episodes=[])
    episode_rows, data_rows = [], []
    cursor = 0
    for sid in range(count):
        n = 3 + sid % 3
        initial = np.array([sid, 1, 2, 3, 4, 5, .02], dtype=np.float32)
        raw_initial = initial.astype(np.float64) * np.array([np.pi / 180] * 6 + [.01])
        manifest["episodes"].append(dict(schema_version="pressb.table_buttons.raw.v1", episode_index=sid,
            frame_count=n, duration_s=(n - 1) / 30, first_success_time_s=(n - 1) / 30 - 1 / 120,
            success=True, stop_reason="success", terminal_action="repeat_previous_interval_target",
            group="near_red", spacing_profile="near", target_color="red", instruction="Press the red button.",
            fps=30, physics_dt=1 / 120, camera_names=["up", "wrist"], camera_resolution=[640, 480],
            state_units=["rad"] * 6 + ["m"], action_units=["rad"] * 6 + ["m"],
            gripper_representation="full_width", seed=sid, initial_state_measured_rad_m=raw_initial.tolist()))
        ep = dict(episode_index=sid, length=n, tasks=["Press the red button."],
                  dataset_from_index=cursor, dataset_to_index=cursor + n,
                  **{"data/chunk_index": 0, "data/file_index": 0})
        for key in buttons.CAMERA_KEYS:
            ep.update({f"videos/{key}/chunk_index": 0, f"videos/{key}/file_index": 0,
                       f"videos/{key}/from_timestamp": cursor / 30,
                       f"videos/{key}/to_timestamp": (cursor + n) / 30})
        episode_rows.append(ep)
        for frame in range(n):
            state = initial.copy()
            state[1] += frame
            action = initial.copy()
            action[1] += min(frame + 1, n - 1)
            data_rows.append(dict(index=cursor + frame, episode_index=sid, frame_index=frame,
                task_index=0, timestamp=np.float32(frame / 30).item(),
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


def test_synthetic_split_chunk_clamp_camera_order_and_strides(synthetic_root, tmp_path):
    split_path = tmp_path / "run/split.json"
    train = buttons.TableButtonsLeRobotDataset(synthetic_root, sample_stride=2, split_manifest=split_path)
    val = buttons.TableButtonsLeRobotDataset(synthetic_root, split="val", split_manifest=split_path)
    train._video_frames = val._video_frames = fake_decode
    assert not set(train.source_episode_ids) & set(val.source_episode_ids)
    assert len(train.source_episode_ids) == 90 and len(val.source_episode_ids) == 10
    assert train.camera_keys == ("observation.images.up", "observation.images.wrist")
    assert train.state_dim == train.action_dim == 7
    assert train.normalization["scale"] == [180.] * 6 + [8.]
    assert train.fingerprint["task"]["group"] == "near_red"
    assert train.fingerprint["source_numeric_sha256"]
    assert "floor" not in train.sample_metadata(0)
    for sid, anchors in train._anchors:
        assert anchors[-1] == train.episode_lengths[sid] - 1
        sample = train.get_frame(sid, 1)
        e = train._episodes[sid]
        final = e["length"] - 1
        rows = np.minimum(np.arange(7) + 1, final) + e["dataset_from_index"]
        np.testing.assert_array_equal(sample["action"], train._actions[rows])
        video_rows = np.minimum(np.arange(8) + 1, final)
        np.testing.assert_array_equal(sample["video_is_pad"], np.arange(8) + 1 > final)
        np.testing.assert_allclose(sample["video_timestamps"], video_rows / 30)
        np.testing.assert_array_equal(sample["video"][0], fake_decode(e["videos"][buttons.CAMERA_KEYS[0]][0],
                                      video_rows + e["dataset_from_index"]))
        assert sample["video"][0, 0, 0, 0, 0] == 10 and sample["video"][1, 0, 0, 0, 0] == 200
        terminal = train.get_frame(sid, final)
        np.testing.assert_array_equal(terminal["action"], np.repeat(terminal["action"][:1], 7, axis=0))
        np.testing.assert_array_equal(terminal["action_is_pad"], [False] + [True] * 6)
        np.testing.assert_array_equal(terminal["video_is_pad"], [False] + [True] * 7)
        assert terminal["lang"] == "Press the red button." and terminal["group"] == "near_red"
    with pytest.raises(KeyError):
        train.get_frame(val.source_episode_ids[0], 0)
    with pytest.raises(IndexError):
        train.get_frame(train.source_episode_ids[0], 999)
    with pytest.raises(ValueError, match="differs"):
        buttons.TableButtonsLeRobotDataset(synthetic_root, split_manifest=split_path, seed=43)
    with pytest.raises(ValueError, match="outside"):
        buttons.TableButtonsLeRobotDataset(synthetic_root, split_manifest=synthetic_root / "split.json")


@pytest.mark.parametrize("bad", ["units", "feature_names", "feature_shape", "unsuccessful", "time", "terminal_contract", "camera_order", "duplicate_seed"])
def test_corrupt_metadata_is_rejected(synthetic_root, bad):
    ip, mp = synthetic_root / "meta/info.json", synthetic_root / "meta/export_manifest.json"
    info, manifest = json.loads(ip.read_text()), json.loads(mp.read_text())
    if bad == "units": manifest["semantics"]["units"][0] = "rad"
    if bad == "feature_names": info["features"]["action"]["names"][0] = "x_m"
    if bad == "feature_shape": info["features"]["action"]["shape"] = [9]
    if bad == "unsuccessful": manifest["episodes"][0]["success"] = False
    if bad == "time": manifest["episodes"][0]["duration_s"] += .1
    if bad == "terminal_contract": manifest["episodes"][0]["terminal_action"] = "zero"
    if bad == "camera_order": manifest["episodes"][0]["camera_names"] = ["wrist", "up"]
    if bad == "duplicate_seed": manifest["episodes"][1]["seed"] = 0
    write_json(ip, info)
    write_json(mp, manifest)
    with pytest.raises(ValueError):
        buttons.TableButtonsLeRobotDataset(synthetic_root)


@pytest.mark.parametrize("bad", ["nonfinite", "timestamp", "frame_index", "task_index", "terminal", "missing"])
def test_corrupt_numeric_is_rejected(synthetic_root, bad):
    path = synthetic_root / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    if bad == "nonfinite": rows[1]["action"][0] = float("nan")
    if bad == "timestamp": rows[1]["timestamp"] = .2
    if bad == "frame_index": rows[1]["frame_index"] = 2
    if bad == "task_index": rows[1]["task_index"] = 1
    if bad == "terminal": rows[2]["action"][0] += 1
    if bad == "missing": rows.pop()
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError):
        buttons.TableButtonsLeRobotDataset(synthetic_root)


@pytest.mark.parametrize("group", GROUPS)
def test_real_all_nine_first_middle_last_episodes_and_frames(group):
    root = DATA_ROOT / group
    if not (root / "meta/export_manifest.json").exists():
        pytest.skip("Set TABLE_BUTTONS_TEST_ROOT to the parent of all nine exported datasets")
    before = (root / "meta/export_manifest.json").read_bytes()
    train = buttons.TableButtonsLeRobotDataset(root, sample_stride=3, frame_cache_size=16, decoder_cache_size=2)
    val = buttons.TableButtonsLeRobotDataset(root, split="val", split_manifest=train.split_manifest,
                                             sample_stride=3, frame_cache_size=16, decoder_cache_size=2)
    try:
        assert len(train.source_episode_ids) == 90 and len(val.source_episode_ids) == 10
        assert set(train.source_episode_ids).isdisjoint(val.source_episode_ids)
        assert list(val.source_episode_ids) == sorted(np.random.default_rng(42).permutation(100)[:10].tolist())
        raw = [r for path in sorted((root / "data").rglob("*.parquet")) for r in pq.read_table(path).to_pylist()]
        raw.sort(key=lambda r: r["index"])
        for sid in (0, 50, 99):
            dataset = train if sid in train.source_episode_ids else val
            episode = dataset._episodes[sid]
            for frame in (0, episode["length"] // 2, episode["length"] - 1):
                sample = dataset.get_frame(sid, frame)
                assert sample["state"].shape == (1, 7) and sample["action"].shape == (7, 7)
                assert sample["video"].shape == (2, 8, 224, 224, 3)
                assert sample["video"].dtype == np.uint8
                assert sample["lang"] == f"Press the {group.split('_')[1]} button."
                index = episode["dataset_from_index"] + frame
                expected_state = np.asarray(raw[index]["observation.state"], np.float32) / np.array([180.] * 6 + [8.], np.float32)
                np.testing.assert_array_equal(sample["state"][0], expected_state)
                np.testing.assert_allclose(buttons.denormalize_joint7(sample["state"][0]),
                                           raw[index]["observation.state"], atol=1e-5, rtol=1e-6)
                for future in range(7):
                    row = raw[episode["dataset_from_index"] + min(frame + future, episode["length"] - 1)]
                    expected = np.asarray(row["action"], np.float32) / np.array([180.] * 6 + [8.], np.float32)
                    np.testing.assert_array_equal(sample["action"][future], expected)
                    np.testing.assert_allclose(buttons.denormalize_joint7(sample["action"][future]),
                                               row["action"], atol=1e-5, rtol=1e-6)
                for view in range(2):
                    np.testing.assert_array_equal(np.asarray(sample["image"][view]), sample["video"][view, 0])
                if frame == episode["length"] - 1:
                    np.testing.assert_array_equal(sample["video_is_pad"], [False] + [True] * 7)
                    np.testing.assert_array_equal(sample["action_is_pad"], [False] + [True] * 6)
                    for view in range(2):
                        np.testing.assert_array_equal(sample["video"][view], np.repeat(sample["video"][view, :1], 8, axis=0))
        # Independently decode episode 0 linearly to check camera/time routing.
        dataset = train if 0 in train.source_episode_ids else val
        sample = dataset.get_frame(0, 0)
        for view, key in enumerate(buttons.CAMERA_KEYS):
            path, offset = dataset._episodes[0]["videos"][key]
            assert offset == 0
            with av.open(path) as container:
                for k, frame in enumerate(container.decode(video=0)):
                    rgb = frame.to_ndarray(format="rgb24")
                    expected = np.asarray(Image.fromarray(rgb).resize((224, 224), Image.Resampling.BILINEAR))
                    np.testing.assert_array_equal(sample["video"][view, k], expected)
                    assert abs(float(frame.pts * frame.time_base) - k / 30) < 1e-6
                    if k == 7: break
        assert (root / "meta/export_manifest.json").read_bytes() == before
        assert len(train._frames) <= 16 and len(train._decoders) <= 2
    finally:
        train.close()
        val.close()
