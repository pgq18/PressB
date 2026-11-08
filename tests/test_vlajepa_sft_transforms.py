"""CPU augmentation checks against every bundled VLA-JEPA SFT reader."""
from __future__ import annotations

from collections import OrderedDict
import importlib
import multiprocessing
import os
from pathlib import Path
import random
import sys
from types import ModuleType

import numpy as np
import pytest


READERS = Path(__file__).resolve().parents[1] / "src/pressb/model_sources/vlajepa/starVLA/dataloader"
PROFILES = {
    "pose9": ("piper_lerobot", "PiperLeRobotDataset"),
    "joint7": ("table_buttons_lerobot", "TableButtonsLeRobotDataset"),
    "joint6": ("table_buttons_joint6_lerobot", "TableButtonsJoint6LeRobotDataset"),
    "joint7_multitask": ("table_buttons_multitask_lerobot", "TableButtonsMultitaskLeRobotDataset"),
    "joint6_multitask": ("table_buttons_multitask_joint6_lerobot", "TableButtonsMultitaskJoint6LeRobotDataset"),
}


@pytest.fixture(scope="module")
def torch_module():
    torch = pytest.importorskip("torch")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield torch
    torch.set_num_threads(previous_threads)


@pytest.fixture(autouse=True)
def preserve_rng(torch_module):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch_module.random.fork_rng(devices=[]):
            torch_module.manual_seed(16384)
            random.seed(16384)
            np.random.seed(16384)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@pytest.fixture(scope="module")
def reader_classes():
    pytest.importorskip("av")
    pytest.importorskip("pyarrow.parquet")
    # Relative reader imports work without importing the model stack or
    # changing the starVLA source selection used by unrelated tests.
    package_name = "_pressb_test_vlajepa_sft_readers"
    package = ModuleType(package_name)
    package.__path__ = [str(READERS)]
    sys.modules[package_name] = package
    try:
        yield {
            profile: getattr(importlib.import_module(f"{package_name}.{module}"), name)
            for profile, (module, name) in PROFILES.items()
        }
    finally:
        for name in tuple(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                del sys.modules[name]


@pytest.fixture(params=PROFILES)
def reader_class(request, reader_classes):
    return reader_classes[request.param]


@pytest.fixture(scope="module")
def sampler_class(torch_module):
    path = READERS.parent / "training/train_piper.py"
    spec = importlib.util.spec_from_file_location("_pressb_test_vlajepa_trainer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OffsetDistributedSampler


@pytest.fixture
def fork_context():
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("Isolated reader packages require fork for these CPU worker tests")
    return "fork"


def _cached_reader(reader_class, image_transforms=None, split="train"):
    """Supply in-memory numeric data and the real reader's decoded-frame cache."""
    reader = reader_class.__new__(reader_class)
    reader.split, reader.image_transforms = split, image_transforms
    reader.action_horizon, reader.video_horizon, reader.fps = 4, 5, 30
    reader.sample_stride, reader.resolution = 1, (24, 26)
    reader.frame_cache_size, reader.decoder_cache_size = 32, 2
    reader.group, reader.target_color, reader.spacing_profile = "near_red", "red", "near"
    reader.instruction = "Press the red button."
    reader._pid = os.getpid()
    reader._frames, reader._decoders = OrderedDict(), OrderedDict()
    reader._anchors, reader._ends = [(73, np.arange(4))], [4]
    episode = dict(length=4, dataset_from_index=2, dataset_to_index=6,
                   episode_index=7, floor=24, task=reader.instruction,
                   target_color="red", task_index=0, original_source_episode_id=13,
                   videos={})
    reader._episodes = {73: episode}
    reader._states = np.arange(6 * reader.state_dim, dtype=np.float32).reshape(6, reader.state_dim) / 100
    reader._actions = np.arange(6 * reader.action_dim, dtype=np.float32).reshape(6, reader.action_dim) / 200
    reader._states.flags.writeable = reader._actions.flags.writeable = False
    for camera_index, key in enumerate(reader.camera_keys):
        path, offset = f"synthetic_camera_{camera_index}.mp4", 11
        episode["videos"][key] = (path, offset)
        # Identical nonuniform frames expose accidental independent jitter
        # within a clip, including its repeated terminal padding frames.
        frame = ((np.arange(24 * 26 * 3).reshape(24, 26, 3) % 151)
                 + 30 + camera_index * 17).astype(np.uint8)
        for index in range(4):
            cached = frame.copy()
            cached.flags.writeable = False
            reader._frames[(path, offset + index)] = cached
    return reader


def _raw_video(reader):
    episode = reader._episodes[73]
    return np.stack([
        np.stack([reader._frames[(path, offset + index)] for index in (2, 3, 3, 3, 3)])
        for path, offset in episode["videos"].values()
    ])


def _assert_sample_contract(sample, reader):
    assert sample["video"].shape == (2, 5, 24, 26, 3)
    assert sample["video"].dtype == np.uint8
    for camera, image in enumerate(sample["image"]):
        assert image.mode == "RGB" and image.size == (26, 24)
        np.testing.assert_array_equal(np.asarray(image), sample["video"][camera, 0])
        for frame in sample["video"][camera, 1:]:
            np.testing.assert_array_equal(frame, sample["video"][camera, 0])
    np.testing.assert_array_equal(sample["state"], reader._states[4:5])
    np.testing.assert_array_equal(sample["action"], reader._actions[[4, 5, 5, 5]])
    np.testing.assert_array_equal(sample["action_is_pad"], [False, False, True, True])
    np.testing.assert_array_equal(sample["video_is_pad"], [False, False, True, True, True])
    np.testing.assert_array_equal(sample["video_timestamps"], np.asarray([2, 3, 3, 3, 3]) / 30)
    assert sample["frame_index"] == 2 and sample["dataset_index"] == 4
    assert sample["source_episode_id"] == 73 and sample["lang"] == reader.instruction


def _rng_snapshot(torch):
    return torch.random.get_rng_state().clone(), random.getstate(), np.random.get_state()


def _assert_rng_unchanged(torch, before):
    assert torch.equal(torch.random.get_rng_state(), before[0])
    assert random.getstate() == before[1]
    actual = np.random.get_state()
    assert actual[0] == before[2][0] and actual[2:] == before[2][2:]
    np.testing.assert_array_equal(actual[1], before[2][1])


def test_default_sft_preset_keeps_images_and_temporal_frames_coherent(reader_class, torch_module):
    pytest.importorskip("torchvision")
    from pressb.training.image_transforms import build_image_transforms

    transform = build_image_transforms(seed=42)
    assert callable(transform)
    reader = _cached_reader(reader_class, transform)
    original = _raw_video(reader)
    before = _rng_snapshot(torch_module)
    sample = reader[2]
    _assert_sample_contract(sample, reader)
    assert not np.array_equal(sample["video"], original)
    # An epoch/sample/camera key survives cache hits, worker restarts and
    # resumed samplers. Epoch changes must still produce fresh augmentation.
    repeated = reader[2]
    _assert_sample_contract(repeated, reader)
    np.testing.assert_array_equal(repeated["video"], sample["video"])
    transform.set_epoch(1)
    next_epoch = reader[2]
    _assert_sample_contract(next_epoch, reader)
    assert not np.array_equal(next_epoch["video"], sample["video"])
    np.testing.assert_array_equal(_raw_video(reader), original)
    _assert_rng_unchanged(torch_module, before)


def test_reader_passes_source_anchor_and_camera_to_keyed_transform(reader_class):
    class KeyedTransform:
        def __init__(self):
            self.keys = []

        def __call__(self, clip):
            pytest.fail("A transform with apply_to_sample must use the keyed path")

        def apply_to_sample(self, clip, sample_key):
            self.keys.append(sample_key)
            return clip

    transform = KeyedTransform()
    reader = _cached_reader(reader_class, transform)
    _assert_sample_contract(reader[2], reader)
    assert transform.keys == [(73, 2, key) for key in reader.camera_keys]


def test_in_place_transform_cannot_mutate_cached_rgb(reader_class, torch_module):
    torch = torch_module
    draws = []

    def in_place_transform(clip):
        assert isinstance(clip, torch.Tensor)
        assert clip.shape == (5, 3, 24, 26) and clip.dtype == torch.uint8
        assert clip.device.type == "cpu"
        offset = int(torch.randint(1, 30, ()).item())
        draws.append(offset)
        return clip.add_(offset)

    reader = _cached_reader(reader_class, in_place_transform)
    cached = {key: value.copy() for key, value in reader._frames.items()}
    original = _raw_video(reader)
    first, second = reader[2], reader[2]
    assert len(draws) == 4
    assert draws[:2] != draws[2:]
    for sample, offsets in ((first, draws[:2]), (second, draws[2:])):
        _assert_sample_contract(sample, reader)
        for camera, offset in enumerate(offsets):
            np.testing.assert_array_equal(sample["video"][camera], original[camera] + offset)
    for key, value in reader._frames.items():
        assert not value.flags.writeable
        np.testing.assert_array_equal(value, cached[key])


@pytest.mark.parametrize("split", ["train", "val"])
def test_unaugmented_and_validation_samples_do_not_consume_rng(reader_class, torch_module, split):
    def forbidden_transform(clip):
        pytest.fail("Validation must not call an augmentation supplied to the reader")

    reader = _cached_reader(reader_class, forbidden_transform if split == "val" else None, split)
    original = _raw_video(reader)
    before = _rng_snapshot(torch_module)
    for _ in range(2):
        sample = reader[2]
        _assert_sample_contract(sample, reader)
        np.testing.assert_array_equal(sample["video"], original)
    _assert_rng_unchanged(torch_module, before)


@pytest.mark.parametrize("split,supplied", [("train", False), ("val", False), ("train", True), ("val", True)])
def test_reader_constructor_defaults_and_validation_suppression(reader_class, tmp_path, split, supplied):
    # Stop at the first metadata read: initialization of the transform option
    # does not require generating thousands of unrelated episode records.
    reader = reader_class.__new__(reader_class)
    marker = lambda clip: clip
    kwargs = {"image_transforms": marker} if supplied else {}
    with pytest.raises(FileNotFoundError, match="info.json"):
        reader.__init__(tmp_path, split=split, **kwargs)
    assert reader.image_transforms is (marker if supplied and split == "train" else None)


class _WorkerCachedReader:
    """Refill synthetic decoded RGB after the real reader clears process caches."""

    def __init__(self, reader):
        self.reader = reader
        self.source_frames = {key: value.copy() for key, value in reader._frames.items()}

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, index):
        self.reader._ensure_process()
        if not self.reader._frames:
            for key, source in self.source_frames.items():
                frame = source.copy()
                frame.flags.writeable = False
                self.reader._frames[key] = frame
        return os.getpid(), self.reader[index]


def _keep_sample(sample):
    return sample


def _loader(torch, reader, sampler, workers, context, *, persistent=False, seed=10):
    kwargs = dict(num_workers=workers, batch_size=None, sampler=sampler,
                  collate_fn=_keep_sample, generator=torch.Generator().manual_seed(seed))
    if workers:
        kwargs.update(multiprocessing_context=context, persistent_workers=persistent,
                      prefetch_factor=2)
    return torch.utils.data.DataLoader(_WorkerCachedReader(reader), **kwargs)


def _sampler(sampler_class, reader, *, epoch, offset=0):
    sampler = sampler_class(reader, num_replicas=1, rank=0, shuffle=True, seed=34)
    sampler.set_epoch(epoch)
    sampler.set_start_index(offset)
    return sampler


def _assert_same_sample(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if key == "image":
            for image, reference in zip(actual[key], value):
                np.testing.assert_array_equal(np.asarray(image), np.asarray(reference))
        elif isinstance(value, np.ndarray):
            np.testing.assert_array_equal(actual[key], value)
        else:
            assert actual[key] == value


@pytest.mark.parametrize("resumed_workers", [0, 1])
def test_worker_restart_and_sampler_offset_resume_identical_samples(
    reader_class, torch_module, sampler_class, fork_context, resumed_workers,
):
    pytest.importorskip("torchvision")
    from pressb.training.image_transforms import build_image_transforms

    transform = build_image_transforms(seed=42)
    transform.set_epoch(7)
    reader = _cached_reader(reader_class, transform)
    sampler = _sampler(sampler_class, reader, epoch=7)
    expected = [reader[index] for index in sampler]
    before = _rng_snapshot(torch_module)
    loader = _loader(torch_module, reader, sampler, 2, fork_context, seed=100)
    iterator = iter(loader)
    try:
        # Workers prefetch beyond the checkpoint, so restarting must neither
        # replay those discarded draws nor alter remaining sample values.
        prefix = [next(iterator), next(iterator)]
    finally:
        iterator._shutdown_workers()
    assert all(pid != os.getpid() for pid, _ in prefix)

    restored_transform = build_image_transforms(seed=42)
    restored_transform.set_epoch(7)
    restored = _cached_reader(reader_class, restored_transform)
    resumed_sampler = _sampler(sampler_class, restored, epoch=7, offset=2)
    resumed = list(_loader(torch_module, restored, resumed_sampler, resumed_workers,
                           fork_context, seed=999))
    assert len(prefix) + len(resumed) == len(expected)
    if resumed_workers:
        assert {pid for pid, _ in prefix}.isdisjoint(pid for pid, _ in resumed)
    for (_, sample), reference in zip(prefix + resumed, expected):
        _assert_same_sample(sample, reference)
    _assert_rng_unchanged(torch_module, before)


def test_persistent_workers_observe_epoch_change_without_restart(
    reader_class, torch_module, sampler_class, fork_context,
):
    pytest.importorskip("torchvision")
    from pressb.training.image_transforms import build_image_transforms

    transform = build_image_transforms(seed=42)
    reader = _cached_reader(reader_class, transform)
    sampler = _sampler(sampler_class, reader, epoch=0)
    loader = _loader(torch_module, reader, sampler, 2, fork_context, persistent=True)
    before = _rng_snapshot(torch_module)
    try:
        first = list(loader)
        transform.set_epoch(1)
        sampler.set_epoch(1)
        second = list(loader)
        assert {pid for pid, _ in first} == {pid for pid, _ in second}
        assert all(pid != os.getpid() for pid, _ in second)
        first_by_anchor = {sample["frame_index"]: sample for _, sample in first}
        assert any(not np.array_equal(sample["video"], first_by_anchor[sample["frame_index"]]["video"])
                   for _, sample in second)

        fresh_transform = build_image_transforms(seed=42)
        fresh_transform.set_epoch(1)
        fresh = _cached_reader(reader_class, fresh_transform)
        assert [sample["frame_index"] for _, sample in second] == list(sampler)
        for _, sample in second:
            _assert_same_sample(sample, fresh[sample["frame_index"]])
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()
    _assert_rng_unchanged(torch_module, before)
