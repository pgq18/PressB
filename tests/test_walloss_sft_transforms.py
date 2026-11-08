"""Exercise Wall-OSS SFT loading with CPU tensors and no model/video dependencies."""

import importlib.util
import json
from copy import deepcopy
from itertools import islice
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1] / "src/pressb/model_sources/walloss"
CAMERAS = ["observation.images.up", "observation.images.wrist"]
BRIGHTNESS = {
    "enable": True,
    "max_num_transforms": 1,
    "random_order": False,
    "tfs": {
        "brightness": {
            "weight": 1.0,
            "type": "ColorJitter",
            "kwargs": {"brightness": [0.5, 0.5]},
        }
    },
}


class FakeMetadata:
    def __init__(self, repo_id, root=None):
        self.repo_id = repo_id
        self.fps = 20
        self.total_episodes = 4
        self.camera_keys = CAMERAS
        self.video_keys = []
        self.tasks = SimpleNamespace(iloc=[SimpleNamespace(name="Press the button")])


class FakeRows:
    def __init__(self, episodes):
        self.indices = [episode * 20 + frame for episode in episodes for frame in range(20)]

    def __getitem__(self, index):
        absolute = self.indices[index]
        return {
            "episode_index": torch.tensor(absolute // 20),
            "index": torch.tensor(absolute),
            "frame_index": torch.tensor(absolute),
            "task_index": torch.tensor(0),
            "timestamp": torch.tensor((absolute % 20) / 20),
            "observation.state": torch.full((7,), absolute / 100),
            "action": torch.full((7,), absolute / 100),
            **{camera: torch.full((3, 16, 16), 0.8) for camera in CAMERAS},
        }

    def __len__(self):
        return len(self.indices)


class FakeLeRobotDataset:
    """Only the storage boundary is fake; FixedLeRobotDataset.__getitem__ runs."""

    def __init__(self, repo_id, *, episodes, delta_timestamps, image_transforms=None, **kwargs):
        self.meta = FakeMetadata(repo_id)
        self.episodes = episodes
        self.num_episodes = len(episodes)
        self.hf_dataset = FakeRows(episodes)
        self.num_frames = len(self.hf_dataset)
        self.image_transforms = image_transforms
        self.delta_indices = {"action": list(range(len(delta_timestamps["action"])))}

    def __len__(self):
        return self.num_frames

    def _ensure_hf_dataset_loaded(self):
        pass

    def _get_query_indices(self, absolute, episode):
        last = (episode + 1) * 20 - 1
        indices = [min(absolute + delta, last) for delta in self.delta_indices["action"]]
        return {"action": indices}, {"action_is_pad": torch.tensor([False] * len(indices))}

    def _query_hf_dataset(self, indices):
        return {"action": torch.tensor(indices["action"])[:, None].expand(-1, 7) / 100}


class FakeBatch(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error


class FakeTokenizer:
    def convert_tokens_to_ids(self, token):
        return 1


def fake_preprocess(*, images, **kwargs):
    pixels = torch.stack([
        torch.stack([torch.from_numpy(np.array(image)).permute(2, 0, 1) for image in views])
        for views in images
    ])
    return FakeBatch(pixel_values=pixels, input_ids=torch.zeros((len(images), 4), dtype=torch.long))


def add_module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def import_module(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def wall_loader(monkeypatch):
    # Stub optional training packages at their I/O boundary. Shared augmentation,
    # frame splitting, FixedLeRobotDataset, preprocessing, and collation are real.
    add_module(monkeypatch, "lerobot", __path__=[])
    add_module(monkeypatch, "lerobot.datasets", __path__=[])
    add_module(monkeypatch, "lerobot.datasets.lerobot_dataset",
               LeRobotDataset=FakeLeRobotDataset, LeRobotDatasetMetadata=FakeMetadata)
    add_module(monkeypatch, "qwen_vl_utils", __path__=[])
    add_module(monkeypatch, "qwen_vl_utils.vision_process", MIN_PIXELS=1,
               MAX_PIXELS=1000000, IMAGE_FACTOR=1,
               smart_resize=lambda height, width, **kwargs: (height, width))
    add_module(monkeypatch, "transformers", AutoProcessor=object)
    add_module(monkeypatch, "wall_x", __path__=[str(ROOT / "wall_x")])
    add_module(monkeypatch, "wall_x.data", __path__=[str(ROOT / "wall_x/data")])
    stats = {key: SimpleNamespace(min=torch.zeros(7), delta=torch.ones(7)) for key in ("action", "state")}
    add_module(
        monkeypatch, "wall_x.data.utils",
        KEY_MAPPINGS={"lerobot/piper": {
            "camera": dict(zip(CAMERAS, ("face_view", "left_wrist_view"))),
            "state": "observation.state", "action": "action",
        }},
        load_norm_stats=lambda *args: stats,
        get_wallx_normal_text=lambda info, *args, **kwargs: (info["instruction"], False),
        process_grounding_points=lambda text, *args: text,
        replace_action_token=lambda text, *args: text,
        preprocesser_call=fake_preprocess,
    )
    import_module(monkeypatch, "wall_x.data.config", ROOT / "wall_x/data/config.py")
    module = import_module(monkeypatch, "wall_x.data.load_lerobot_dataset",
                           ROOT / "wall_x/data/load_lerobot_dataset.py")

    def load_processor(collator):
        collator.processor = SimpleNamespace(tokenizer=FakeTokenizer())
        collator.train_action_tokenizer = None

    monkeypatch.setattr(module.DataCollator, "load_processor", load_processor)
    return module


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    # Avoid inheriting a live CPU thread pool when persistent workers fork.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def configs(num_workers=0):
    lerobot_config = {"repo_id": "lerobot/piper"}
    config = {
        "norm_stats_path": "unused.json",
        "batch_size_per_gpu": 1,
        "num_workers": num_workers,
        "image_transforms": BRIGHTNESS,
        "data": {
            "train_test_split": 0.75,
            "split_seed": 17,
            "action_horizon": 2,
            "predict_action_keys": [],
            "obs_action_keys": [],
            "resolution": {"face_view": -1, "left_wrist_view": -1},
            "lerobot_config": lerobot_config,
        },
    }
    return config, lerobot_config


def collect_batches(loader):
    result = {}
    for batch in loader:
        absolute = batch["frame_index"].item()
        result[absolute] = batch["pixel_values"].clone()
        expected_action = torch.tensor([absolute, min(absolute + 1, (absolute // 20 + 1) * 20 - 1)]) / 100 * 2 - 1
        torch.testing.assert_close(batch["action_chunk"][0, :, :7], expected_action[:, None].expand(-1, 7))
        torch.testing.assert_close(batch["proprioception"][0, 0, :7], torch.full((7,), absolute / 100 * 2 - 1))
        assert batch["action_chunk"].shape == (1, 2, 20)
        assert batch["dof_mask"][..., :7].all()
        assert not batch["dof_mask"][..., 7:].any()
    return result


def stop_workers(*loaders):
    for loader in loaders:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()


@pytest.mark.parametrize("num_workers", [0, 1])
@pytest.mark.parametrize("validation_first", [False, True])
def test_train_and_validation_iteration_stay_isolated(wall_loader, num_workers, validation_first):
    config, lerobot_config = configs(num_workers)
    dataset, train_num = wall_loader.load_lerobot_data(config, lerobot_config, seed=11)
    expected_train, expected_val = torch.utils.data.random_split(
        range(60), [0.95, 0.05], generator=torch.Generator().manual_seed(11)
    )
    assert dataset.train_dataset.indices == expected_train.indices
    assert dataset.val_dataset.indices == expected_val.indices
    assert train_num == 57
    assert dataset.train_dataset.dataset is not dataset.val_dataset.dataset
    assert dataset.train_dataset.dataset.hf_dataset is dataset.val_dataset.dataset.hf_dataset
    assert dataset.train_dataset.dataset.meta is dataset.val_dataset.dataset.meta
    assert dataset.val_dataset.dataset.image_transforms is None

    if validation_first:
        val_loader, _ = dataset.get_val_dataloader()
        train_loader, train_sampler = dataset.get_train_dataloader()
    else:
        train_loader, train_sampler = dataset.get_train_dataloader()
        val_loader, _ = dataset.get_val_dataloader()
    assert train_loader.dataset is not val_loader.dataset
    assert train_loader.sampler.dataset is train_loader.dataset
    assert val_loader.sampler.dataset is val_loader.dataset
    try:
        dataset._eval()
        first_train = collect_batches(train_loader)
        dataset._train()
        first_val = collect_batches(val_loader)
        train_sampler.set_epoch(1)
        dataset._eval()
        second_train = collect_batches(train_loader)
        dataset._train()
        second_val = collect_batches(val_loader)
        assert len(first_train) == 57 and len(first_val) == 3
        assert first_train.keys().isdisjoint(first_val)
        assert first_train.keys() == second_train.keys()
        assert first_val.keys() == second_val.keys()
        for absolute, pixels in first_train.items():
            assert (pixels == 102).all()
            torch.testing.assert_close(pixels, second_train[absolute])
        for absolute, pixels in first_val.items():
            assert (pixels == 204).all()
            torch.testing.assert_close(pixels, second_val[absolute])
        selected_episodes = set(dataset.train_episodes)
        assert {absolute // 20 for absolute in first_train | first_val} == selected_episodes
        assert selected_episodes.isdisjoint(dataset.test_episodes)
    finally:
        stop_workers(train_loader, val_loader)


@pytest.mark.parametrize("root_value,legacy_value,enabled", [
    ("absent", "absent", True),
    (None, {"enable": False}, True),
    ("absent", {"enable": False}, False),
    ({"enable": False}, BRIGHTNESS, False),
    ("absent", BRIGHTNESS, True),
])
def test_loader_default_and_legacy_configuration(wall_loader, root_value, legacy_value, enabled):
    config, lerobot_config = configs()
    if root_value == "absent":
        config.pop("image_transforms")
    else:
        config["image_transforms"] = root_value
    if legacy_value != "absent":
        lerobot_config["image_transforms"] = legacy_value
    dataset, _ = wall_loader.load_lerobot_data(config, lerobot_config)
    assert callable(dataset.hf_dataset.image_transforms) == enabled
    assert dataset.val_dataset.dataset.image_transforms is None
    if not enabled:
        train_loader, _ = dataset.get_train_dataloader()
        assert all((pixels == 204).all() for pixels in collect_batches(train_loader).values())


def test_independent_test_dataset_never_augments(wall_loader):
    config, lerobot_config = configs()
    lerobot_config["image_transforms"] = BRIGHTNESS
    dataset = wall_loader.load_test_dataset(config, lerobot_config, episode=2)
    assert dataset.hf_dataset.image_transforms is None
    first = collect_batches(dataset.get_dataloader())
    second = collect_batches(dataset.get_dataloader())
    assert list(first) == list(range(40, 60))
    for absolute, pixels in first.items():
        assert (pixels == 204).all()
        torch.testing.assert_close(pixels, second[absolute])


@pytest.mark.parametrize("num_workers", [0, 1])
def test_keyed_augmentation_tracks_epoch_and_survives_worker_restart(wall_loader, num_workers):
    config, lerobot_config = configs(num_workers)
    config["image_transforms"] = deepcopy(BRIGHTNESS)
    config["image_transforms"]["tfs"]["brightness"]["kwargs"]["brightness"] = [0.25, 1.0]
    dataset, _ = wall_loader.load_lerobot_data(config, lerobot_config, seed=23)
    train_loader, sampler = dataset.get_train_dataloader()
    restarted_loader = None
    try:
        dataset.set_epoch(0)
        first = collect_batches(train_loader)
        dataset._eval()
        dataset.set_epoch(3)
        sampler.set_epoch(3)
        next_epoch = collect_batches(train_loader)
        assert any(not torch.equal(pixels, next_epoch[absolute]) for absolute, pixels in first.items())
        # Persistent workers see the epoch update, including a return to epoch 0.
        dataset.set_epoch(0)
        sampler.set_epoch(0)
        repeated = collect_batches(train_loader)
        for absolute, pixels in first.items():
            torch.testing.assert_close(pixels, repeated[absolute])

        # Reconstruct the data pipeline with a different worker count and skip a
        # batch prefix. Neither worker RNG state nor preceding reads affect keys.
        config["num_workers"] = 1 - num_workers
        restarted, _ = wall_loader.load_lerobot_data(config, lerobot_config, seed=23)
        restarted.set_epoch(3)
        restarted_loader, restarted_sampler = restarted.get_train_dataloader()
        restarted_sampler.set_epoch(3)
        torch.rand(100)
        remaining = collect_batches(islice(restarted_loader, 13, None))
        assert list(remaining) == list(next_epoch)[13:]
        for absolute, pixels in remaining.items():
            torch.testing.assert_close(pixels, next_epoch[absolute])
    finally:
        stop_workers(train_loader, restarted_loader)


def test_existing_transform_callables_remain_supported(wall_loader):
    config, lerobot_config = configs()
    dataset, _ = wall_loader.load_lerobot_data(config, lerobot_config)
    dataset.hf_dataset.image_transforms = lambda image: image * 0.5
    dataset.set_epoch(4)
    train_loader, _ = dataset.get_train_dataloader()
    assert all((pixels == 102).all() for pixels in collect_batches(train_loader).values())


def test_native_config_records_resolved_defaults_before_logging(monkeypatch, tmp_path):
    import yaml

    add_module(monkeypatch, "wandb", init=lambda **kwargs: "logger")
    add_module(monkeypatch, "accelerate", Accelerator=object,
               DistributedDataParallelKwargs=object, DataLoaderConfiguration=object)
    add_module(monkeypatch, "wall_x", __path__=[])
    add_module(monkeypatch, "wall_x.trainer", __path__=[])
    add_module(monkeypatch, "wall_x.trainer.qwen_vl_act_trainer", QwenVlAct_Trainer=object)
    module = import_module(monkeypatch, "walloss_train_config_test", ROOT / "train_qact.py")
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump({"model_type": "qwen2_5", "data": {}}))
    resolved = module.load_config(path)
    assert resolved["image_transforms"]["enable"] is True
    assert resolved["image_transforms"]["max_num_transforms"] == 3
    assert set(resolved["image_transforms"]["tfs"]) == {
        "brightness", "contrast", "saturation", "hue", "sharpness", "affine"
    }
    save_path = tmp_path / "run"
    resolved.update(save_path=str(save_path), log_project="test", log_name="test")
    assert module.setup_logging(resolved, SimpleNamespace(is_main_process=False)) is None
    assert not save_path.exists()
    assert module.setup_logging(resolved, SimpleNamespace(is_main_process=True)) == "logger"
    snapshot = save_path / "resolved_training_config.json"
    assert json.loads(snapshot.read_text()) == resolved
    # A resumed run records new settings without losing the original snapshot.
    resolved["image_transforms"]["enable"] = False
    module.setup_logging(resolved, SimpleNamespace(is_main_process=True))
    assert json.loads(snapshot.read_text())["image_transforms"]["enable"] is False
    previous, = save_path.glob("resolved_training_config.previous-*.json")
    assert json.loads(previous.read_text())["image_transforms"]["enable"] is True
    module.setup_logging(resolved, SimpleNamespace(is_main_process=True))
    assert len(list(save_path.glob("resolved_training_config.previous-*.json"))) == 1
