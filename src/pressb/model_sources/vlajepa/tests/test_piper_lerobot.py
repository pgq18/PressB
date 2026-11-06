"""Piper pose conventions, episode splits, v3 video offsets and padding.

Set PIPER_TEST_ROOT to the published dataset for integration tests. The local
PressB workspace location is detected as a convenience; otherwise they skip.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys

import av
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import pytest


# Keep this adapter's four-library dependency contract separate from the
# repository package __init__, which initializes unrelated model/data modules.
MODULE = Path(__file__).resolve().parents[1] / "starVLA/dataloader/piper_lerobot.py"
SPEC = importlib.util.spec_from_file_location("piper_lerobot_native", MODULE)
piper = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = piper
SPEC.loader.exec_module(piper)
DEFAULT_ROOT = Path(__file__).resolve().parents[4] / "datasets/piper_elevator_lerobot_press_30hz"
DATA_ROOT = Path(os.environ.get("PIPER_TEST_ROOT", str(DEFAULT_ROOT)))


def test_pose9_preserves_xyz_and_uses_rows_not_columns():
    # Active +90-degree rotation about Z: first row is [0,-1,0].
    q = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
    source = np.array([.123456789, -.7, .31, *q, .008], dtype=np.float32)
    actual = piper.pose8_to_pose9(source)
    assert actual.dtype == source.dtype and actual.shape == (9,)
    np.testing.assert_array_equal(actual[:3], source[:3])
    np.testing.assert_allclose(actual[3:], [0, -1, 0, 1, 0, 0], atol=1e-7)
    changed_width = source.copy()
    changed_width[-1] = .07
    np.testing.assert_array_equal(piper.pose8_to_pose9(changed_width), actual)
    source64 = source.astype(np.float64)
    source64[0] = .123456789123456789
    np.testing.assert_array_equal(piper.pose8_to_pose9(source64)[:3], source64[:3])


def test_rotation_roundtrip_covers_quaternion_sign_and_pi():
    q = np.random.default_rng(5).normal(size=(1000, 4))
    q[:4] = np.eye(4)  # identity and three pi rotations exercise all branches.
    matrix = piper.quaternion_wxyz_to_matrix(q)
    rot6 = piper.matrix_to_rotation_6d(matrix)
    rebuilt = piper.rotation_6d_to_matrix(rot6)
    np.testing.assert_allclose(rebuilt, matrix, atol=2e-14)
    np.testing.assert_allclose(piper.quaternion_wxyz_to_matrix(-q), matrix, atol=1e-14)
    inverse_q = piper.rotation_6d_to_quaternion_wxyz(rot6)
    np.testing.assert_allclose(np.linalg.norm(inverse_q, axis=-1), 1., atol=1e-14)
    np.testing.assert_allclose(piper.quaternion_wxyz_to_matrix(inverse_q), matrix, atol=2e-14)
    np.testing.assert_allclose(np.linalg.det(rebuilt), 1., atol=1e-14)


@pytest.mark.parametrize("bad", [np.zeros(6), [1, 0, 0, 2, 0, 0], [1, 0, 0, np.nan, 1, 0]])
def test_degenerate_rot6_is_rejected(bad):
    with pytest.raises(ValueError):
        piper.rotation_6d_to_matrix(bad)


def test_bad_quaternion_and_nonrotation_matrix_rejected():
    with pytest.raises(ValueError):
        piper.quaternion_wxyz_to_matrix([0, 0, 0, 0])
    with pytest.raises(ValueError):
        piper.matrix_to_quaternion_wxyz(np.diag([1, 1, -1]))


def test_split_is_stratified_deterministic_order_independent_and_bound(tmp_path):
    entries = [dict(episode_id=i, floor=24+i % 12) for i in range(1200)]
    split = piper._build_split(entries, "source_digest", 42)
    assert split == piper._build_split(list(reversed(entries)), "source_digest", 42)
    train, val = set(split["train_source_episode_ids"]), set(split["val_source_episode_ids"])
    assert len(train) == 1080 and len(val) == 120 and not train & val
    assert train | val == set(range(1200))
    for floor in range(12):
        assert sum(i % 12 == floor for i in train) == 90
        assert sum(i % 12 == floor for i in val) == 10
    assert split != piper._build_split(entries, "source_digest", 43)
    root = tmp_path / "dataset"
    root.mkdir()
    path = tmp_path / "run/split.json"
    assert piper._split_manifest(split, path, root) == split
    before = path.read_bytes()
    assert piper._split_manifest(split, path, root) == split and path.read_bytes() == before
    changed = copy.deepcopy(split)
    changed["source_export_manifest_sha256"] = "another_dataset"
    with pytest.raises(ValueError):
        piper._split_manifest(changed, path, root)
    changed = copy.deepcopy(split)
    changed["train_source_episode_ids"][0] = changed["val_source_episode_ids"][0]
    with pytest.raises(ValueError):
        piper._split_manifest(split, changed, root)
    with pytest.raises(ValueError, match="outside"):
        piper._split_manifest(split, root / "split.json", root)


@pytest.fixture(scope="module")
def datasets():
    if not (DATA_ROOT / "meta/export_manifest.json").exists():
        pytest.skip("Set PIPER_TEST_ROOT to the real published press_30hz dataset")
    train = piper.PiperLeRobotDataset(DATA_ROOT, split="train", sample_stride=3)
    val = piper.PiperLeRobotDataset(DATA_ROOT, split="val", split_manifest=train.split_manifest)
    yield train, val
    train.close()
    val.close()


def original_rows(dataset, sid):
    e = dataset._episodes[sid]
    path = dataset.root / dataset.info["data_path"].format(chunk_index=e["data/chunk_index"], file_index=e["data/file_index"])
    rows = pq.read_table(path, columns=["episode_index", "frame_index", "observation.state", "action"]).to_pylist()
    return [r for r in rows if r["episode_index"] == e["episode_index"]]


def test_real_numeric_geometry_split_and_stride(datasets):
    train, val = datasets
    assert train.state_dim == train.action_dim == 9 and train.camera_keys == piper.CAMERA_KEYS
    assert len(train.source_episode_ids) == 1080 and len(val.source_episode_ids) == 120
    assert not set(train.source_episode_ids) & set(val.source_episode_ids)
    sid = train.source_episode_ids[0]
    rows = original_rows(train, sid)
    episode = train._episodes[sid]
    start = episode["dataset_from_index"]
    for index in [0, len(rows)//2, len(rows)-1]:
        for raw_key, loaded in (("observation.state", train._states), ("action", train._actions)):
            original = np.asarray(rows[index][raw_key], np.float32)
            np.testing.assert_array_equal(loaded[start+index, :3], original[:3])
            np.testing.assert_allclose(piper.rotation_6d_to_matrix(loaded[start+index, 3:]),
                                       piper.quaternion_wxyz_to_matrix(original[3:7]), atol=6e-8)
    # Stride affects anchors, never temporal targets. Non-anchor frame 1 remains accessible.
    assert train.sample_metadata(0)["frame_index"] == 0 and train.sample_metadata(1)["frame_index"] == 3
    assert train.get_frame(sid, 1)["frame_index"] == 1
    with pytest.raises(KeyError):
        train.get_frame(val.source_episode_ids[0], 0)
    for source, anchors in train._anchors:
        assert anchors[-1] == train.episode_lengths[source] - 1


def test_real_terminal_shapes_clamp_and_padding(datasets):
    train, _ = datasets
    sid = train.source_episode_ids[-1]
    final = train.episode_lengths[sid]-1
    sample = train.get_frame(sid, final)
    assert sample["state"].shape == (1, 9) and sample["action"].shape == (7, 9)
    assert sample["state"].dtype == sample["action"].dtype == np.float32
    assert sample["video"].shape == (2, 8, 224, 224, 3) and sample["video"].dtype == np.uint8
    assert len(sample["image"]) == 2 and all(isinstance(v, Image.Image) and v.size == (224, 224) for v in sample["image"])
    np.testing.assert_array_equal(sample["action_is_pad"], [False]+[True]*6)
    np.testing.assert_array_equal(sample["video_is_pad"], [False]+[True]*7)
    np.testing.assert_array_equal(sample["action"], np.repeat(sample["action"][:1], 7, axis=0))
    for view in range(2):
        np.testing.assert_array_equal(sample["video"][view], np.repeat(sample["video"][view, :1], 8, axis=0))
        np.testing.assert_array_equal(np.asarray(sample["image"][view]), sample["video"][view, 0])
    before = train.get_frame(sid, final-2)
    np.testing.assert_array_equal(before["action_is_pad"], [False]*3+[True]*4)
    np.testing.assert_array_equal(before["video_is_pad"], [False]*3+[True]*5)
    np.testing.assert_array_equal(before["action"][-1], sample["action"][0])
    assert sample["lang"] == f"Press {sample['floor']} floor."


def test_real_nonzero_video_offset_matches_independent_linear_decode(datasets):
    train, val = datasets
    # Episode 1 shares a file with episode 0; its first image starts at 8.3s.
    dataset = train if 1 in train.source_episode_ids else val
    sample = dataset.get_frame(1, 0)
    e = dataset._episodes[1]
    for view, key in enumerate(piper.CAMERA_KEYS):
        path, first = e["videos"][key]
        assert first > 0
        expected = []
        with av.open(path) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            for frame in container.decode(stream):
                index = int(round(float(frame.pts*frame.time_base)*30))
                if first <= index < first+8:
                    rgb = frame.to_ndarray(format="rgb24")
                    expected.append(np.asarray(Image.fromarray(rgb).resize((224,224), Image.Resampling.BILINEAR)))
                if index >= first+7:
                    break
        np.testing.assert_array_equal(sample["video"][view], np.stack(expected))
    assert not sample["action_is_pad"].any() and not sample["video_is_pad"].any()
    np.testing.assert_allclose(sample["video_timestamps"], np.arange(8)/30)


def test_cache_does_not_alias_samples_and_pickling_drops_decoders(datasets):
    train, _ = datasets
    sample = train[0]
    saved = sample["video"].copy()
    sample["video"][:] = 0
    sample["action"][:] = 0
    repeated = train[0]
    np.testing.assert_array_equal(repeated["video"], saved)
    assert np.any(repeated["action"])
    assert not train.__getstate__()["_decoders"] and not train.__getstate__()["_frames"]
    restored = pickle.loads(pickle.dumps(train))
    try:
        np.testing.assert_array_equal(restored[0]["video"], saved)
    finally:
        restored.close()


def test_real_multiworker_dataloader(datasets):
    torch = pytest.importorskip("torch")
    train, _ = datasets
    indices = [0, 1, len(train)-1]
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train, indices), batch_size=2,
                                        num_workers=2, collate_fn=piper.collate_fn)
    actual = [sample for batch in loader for sample in batch]
    assert len(actual) == len(indices)
    for index, sample in zip(indices, actual):
        metadata = train.sample_metadata(index)
        assert sample["source_episode_id"] == metadata["source_episode_id"]
        assert sample["frame_index"] == metadata["frame_index"]
        expected = train[index]
        np.testing.assert_array_equal(sample["action"], expected["action"])
        np.testing.assert_array_equal(sample["video"], expected["video"])
