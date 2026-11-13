from types import SimpleNamespace
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import json

import pytest

torch = pytest.importorskip("torch")

from pressb.training.pi05_data import ACTION, STATE, PI05Dataset, episode_split, wire_metadata


def test_pi05_registered_supervised_entrypoint(tmp_path, capsys):
    from pressb.training.cli import main

    config = tmp_path / "pi05.yaml"
    config.write_text("{}\n")
    assert main(["--model", "pi05", "--config", str(config), "--dry-run", "--", "--prepare-only"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["module"] == "pressb.training.pi05"
    assert report["argv"][1:] == ["--config", str(config), "--prepare-only"]
    assert report["launches_training"] is False


def test_pi05_policy_overrides_a_saved_native_config(tmp_path):
    pytest.importorskip("lerobot")
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from pressb.training.pi05 import policy_config

    PI05Config(device="cpu").save_pretrained(tmp_path)
    policy, checkpoint = policy_config(dict(type="pi05", pretrained_path=str(tmp_path), device="cpu",
                                            chunk_size=7, n_action_steps=7))
    assert checkpoint == tmp_path
    assert policy.chunk_size == policy.n_action_steps == 7


def test_pi05_per_device_batch_size_is_explicit_and_backward_compatible():
    from pressb.training.pi05 import _per_device_batch_size

    assert _per_device_batch_size({}) == 4
    assert _per_device_batch_size({"batch_size": 8}) == 8
    assert _per_device_batch_size({"per_device_batch_size": 16}) == 16
    assert _per_device_batch_size({"batch_size": 16, "per_device_batch_size": 16}) == 16
    with pytest.raises(ValueError, match="must agree"):
        _per_device_batch_size({"batch_size": 32, "per_device_batch_size": 16})


def test_pi05_torchrun_uses_local_rank_and_destroys_process_group(monkeypatch):
    import torch.distributed as dist
    import pressb.training.pi05 as training

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    actions = []
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: actions.append(("device", str(device))))
    monkeypatch.setattr(dist, "init_process_group", lambda **kwargs: actions.append(("backend", kwargs["backend"])))
    monkeypatch.setattr(dist, "destroy_process_group", lambda: actions.append(("destroy",)))

    def mock_train(config, *, rank, world_size, local_device):
        assert config["policy"]["device"] == "cuda:0"
        actions.append(("train", rank, world_size, str(local_device)))
        raise RuntimeError("model test exception")

    monkeypatch.setattr(training, "_train", mock_train)
    with pytest.raises(RuntimeError, match="model test exception"):
        training.train({"policy": {"device": "cuda:0"}})
    assert actions == [("device", "cuda:1"), ("backend", "nccl"), ("train", 1, 2, "cuda:1"), ("destroy",)]


def test_pi05_split_uses_distinct_episodes_and_keeps_the_same_seed():
    split = episode_split(20, [2, 4, 7, 11], .25, 42)
    assert split == episode_split(20, [11, 7, 4, 2], .25, 42)
    assert set(split["train"]).isdisjoint(split["val"])
    assert sorted(split["train"] + split["val"]) == [2, 4, 7, 11]
    with pytest.raises(ValueError, match="distinct"):
        episode_split(20, [2, 2], .25, 42)


def test_pi05_training_augmentations_are_sample_keyed_and_validation_is_clean():
    camera = "observation.images.up"
    image = torch.full((3, 12, 12), 112, dtype=torch.uint8)
    item = {STATE: torch.zeros(2), ACTION: torch.zeros(3, 2), "task": "Stack the cubes",
            "episode_index": 1, "index": 43, camera: image}
    config = {"enable": True, "tfs": {
        "brightness": {"type": "ColorJitter", "kwargs": {"brightness": [0.4, 0.4]}},
    }}
    train = PI05Dataset([item], [camera], config, seed=9, split="train")
    val = PI05Dataset([item], [camera], config, seed=9, split="val")
    assert torch.equal(val[0][camera], image)
    assert torch.equal(train[0][camera], train[0][camera])
    assert not torch.equal(train[0][camera], image)
    assert torch.all(image == 112)
    assert torch.equal(train[0][ACTION], val[0][ACTION])


def test_pi05_wire_metadata_validates_dataset_fps_and_coordinate_scales():
    camera = "observation.images.up"
    source = SimpleNamespace(
        camera_keys=[camera],
        features={STATE: {"shape": [2]}, ACTION: {"shape": [2]}},
        fps=30,
    )
    spec = {"names": ["joint1", "gripper_width"], "units": ["rad", "m"]}
    wire = {"observation": {"state": spec, "cameras": ["up"]},
            "camera_keys": {"up": camera},
            "action": {**spec, "representation": "joint_position", "frame": "robot_base", "control_hz": 30},
            "state_scale": [57.3, 100], "action_scale": [0.01745, .01]}
    assert wire_metadata(source, wire)["state_scale"] == [57.3, 100.]
    assert wire_metadata(source, wire)["action_scale"] == [0.01745, .01]
    with pytest.raises(ValueError, match="FPS"):
        wire_metadata(source, {**wire, "action": {**wire["action"], "control_hz": 20}})
    with pytest.raises(ValueError, match="factor"):
        wire_metadata(source, {**wire, "state_scale": [1, 0]})


def test_pi05_checkpoint_retention_keeps_the_best_and_two_recent(tmp_path, monkeypatch):
    import pressb.training.pi05 as training

    def save(folder):
        folder.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(training, "version", lambda _: "0.6.1")
    model = SimpleNamespace(save_pretrained=save)
    tokenizer = SimpleNamespace(input_tokenizer=model)
    preprocessor = SimpleNamespace(save_pretrained=save, steps=[tokenizer])
    for step in range(1, 5):
        folder = training._save_checkpoint(tmp_path, step, model, preprocessor, model, {}, best=step == 1, keep=2)
        assert (folder / "pressb_pi05.json").is_file()
        assert (folder / "tokenizer").is_dir()
    checkpoints = tmp_path / "checkpoints"
    assert sorted(p.name for p in checkpoints.glob("step_*")) == ["step_00000001", "step_00000003", "step_00000004"]
    assert (checkpoints / "best").resolve().name == "step_00000001"
    assert (checkpoints / "last").resolve().name == "step_00000004"


def test_pi05_accumulation_requires_a_positive_integer():
    from pressb.training.pi05 import _accumulation_steps

    assert _accumulation_steps({}) == 1
    assert _accumulation_steps({"gradient_accumulation_steps": 4}) == 4
    for value in (0, -1, True, 1.5, "2"):
        with pytest.raises(ValueError, match="positive integer"):
            _accumulation_steps({"gradient_accumulation_steps": value})


class _RegressionPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, batch):
        return (self.linear(batch["input"]) - batch["action"]).square().mean(), {}


def test_pi05_accumulated_update_matches_full_batch_including_short_tail():
    from pressb.training.pi05 import _optimizer_step

    torch.manual_seed(114)
    complete = _RegressionPolicy()
    accumulated = deepcopy(complete)
    raw = {"input": torch.randn(5, 2), "action": torch.randn(5, 1)}
    microbatches = [{key: value[start:end] for key, value in raw.items()}
                    for start, end in ((0, 2), (2, 4), (4, 5))]
    outcomes = []
    for model, batches in ((complete, [raw]), (accumulated, microbatches)):
        optimizer = torch.optim.SGD(model.parameters(), lr=.03)
        outcomes.append(_optimizer_step(
            model, optimizer, batches, lambda batch: batch, [], torch.device("cpu"),
            bf16=False, parameters=list(model.parameters()), max_grad_norm=.7, world_size=1,
        ))
    assert outcomes[0][2] == outcomes[1][2] == 5
    torch.testing.assert_close(outcomes[0][0], outcomes[1][0])
    torch.testing.assert_close(outcomes[0][1], outcomes[1][1])
    for expected, actual in zip(complete.parameters(), accumulated.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)


def test_pi05_ddp_accumulation_synchronizes_only_the_last_microbatch(monkeypatch):
    import torch.distributed as dist
    from pressb.training.pi05 import _optimizer_step

    class DDPStub(_RegressionPolicy):
        def __init__(self):
            super().__init__()
            self.sync_enabled = True
            self.forward_sync, self.backward_sync = [], []
            self.linear.weight.register_hook(lambda gradient: self.backward_sync.append(self.sync_enabled))

        @contextmanager
        def no_sync(self):
            self.sync_enabled = False
            try:
                yield
            finally:
                self.sync_enabled = True

        def forward(self, batch):
            self.forward_sync.append(self.sync_enabled)
            return super().forward(batch)

    model = DDPStub()
    batches = [{"input": torch.ones(2, 2), "action": torch.ones(2, 1)} for _ in range(3)]
    monkeypatch.setattr(dist, "all_reduce", lambda tensor: tensor.mul_(2))
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    _optimizer_step(model, optimizer, batches, lambda batch: batch, [], torch.device("cpu"),
                    bf16=False, parameters=list(model.parameters()), max_grad_norm=1., world_size=2)
    assert model.forward_sync == model.backward_sync == [False, False, True]


def test_pi05_nonfinite_accumulation_does_not_update_weights():
    from pressb.training.pi05 import _optimizer_step

    model = _RegressionPolicy()
    original = deepcopy(model.state_dict())
    batches = [{"input": torch.ones(2, 2), "action": torch.full((2, 1), value)}
               for value in (1., float("nan"))]
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        _optimizer_step(model, optimizer, batches, lambda batch: batch, [], torch.device("cpu"),
                        bf16=False, parameters=list(model.parameters()), max_grad_norm=1., world_size=1)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name])


def test_pi05_accumulation_cycles_epochs_and_rejects_an_empty_loader():
    from pressb.training.pi05 import _training_batches

    data_epochs, sampler_epochs = [], []
    data = SimpleNamespace(set_epoch=data_epochs.append)
    sampler = SimpleNamespace(set_epoch=sampler_epochs.append)
    iterator = _training_batches(data, ["first", "last"], sampler)
    assert [next(iterator) for _ in range(5)] == [
        ("first", 0), ("last", 0), ("first", 1), ("last", 1), ("first", 2),
    ]
    assert data_epochs == sampler_epochs == [0, 1, 2]
    with pytest.raises(ValueError, match="no batches"):
        next(_training_batches(data, [], None))


def test_pi05_npu_torchrun_maps_visible_device_and_uses_hccl(monkeypatch):
    import torch.distributed as dist
    import pressb.models.pi05 as adapter
    import pressb.training.pi05 as training

    class Device:
        type, index = "npu", 2

        def __str__(self):
            return "npu:2"

    actions = []
    for key, value in (("WORLD_SIZE", "4"), ("RANK", "2"), ("LOCAL_RANK", "2")):
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(adapter, "initialize_pi05_device", lambda value: actions.append(("init", value)) or Device())
    monkeypatch.setattr(torch, "npu", SimpleNamespace(
        device_count=lambda: 4, set_device=lambda device: actions.append(("device", str(device))),
    ), raising=False)
    monkeypatch.setattr(dist, "init_process_group", lambda **kwargs: actions.append(("backend", kwargs["backend"])))
    monkeypatch.setattr(dist, "destroy_process_group", lambda: actions.append(("destroy",)))
    monkeypatch.setattr(training, "_train", lambda config, **kwargs: (kwargs["rank"], kwargs["world_size"]))
    assert training.train({"policy": {"device": "npu:0"}}) == (2, 4)
    assert actions == [("init", "npu:2"), ("device", "npu:2"), ("backend", "hccl"), ("destroy",)]


def test_pi05_npu_validation_uses_device_rng_and_bf16_autocast(monkeypatch):
    from pressb.training.pi05 import validate

    calls = []
    device = SimpleNamespace(type="npu", index=3)
    monkeypatch.setattr(torch.random, "fork_rng", lambda **kwargs: calls.append(("rng", kwargs)) or nullcontext())
    monkeypatch.setattr(torch, "autocast", lambda **kwargs: calls.append(("amp", kwargs)) or nullcontext())
    monkeypatch.setattr(torch, "npu", SimpleNamespace(
        device=lambda value: calls.append(("device", value.index)) or nullcontext(),
        manual_seed=lambda seed: calls.append(("seed", seed)),
    ), raising=False)
    model = _RegressionPolicy()
    batch = {"input": torch.ones(2, 2), "action": torch.ones(2, 1)}
    assert validate(model, [batch], lambda value: value, [], device, 91, bf16=True) >= 0
    assert calls == [("rng", {"devices": [3], "device_type": "npu"}),
                     ("amp", {"device_type": "npu", "dtype": torch.bfloat16, "enabled": True}),
                     ("device", 3), ("seed", 91)]


def test_pi05_validation_preserves_cpu_training_rng():
    from pressb.training.pi05 import validate

    class RandomPolicy(torch.nn.Module):
        def forward(self, batch):
            return torch.rand(()), {}

    loader = [{"action": torch.zeros(2, 16, 1)}]
    torch.manual_seed(39)
    initial = torch.random.get_rng_state().clone()
    first = validate(RandomPolicy(), loader, lambda batch: batch, [], torch.device("cpu"), 17, bf16=False)
    assert torch.equal(initial, torch.random.get_rng_state())
    second = validate(RandomPolicy(), loader, lambda batch: batch, [], torch.device("cpu"), 17, bf16=False)
    assert first == second


def test_pi05_accumulation_counts_optimizer_steps_and_logs_effective_batch(tmp_path, monkeypatch):
    import pressb.models.pi05 as adapter
    import pressb.training.pi05 as training

    class Data(torch.utils.data.Dataset):
        def set_epoch(self, epoch):
            pass

        def __len__(self):
            return 5

        def __getitem__(self, index):
            return {"input": torch.ones(2), "action": torch.ones(1)}

    model = _RegressionPolicy()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    updates, scheduler_steps, checkpoints = [], [], []
    optimizer.register_step_post_hook(lambda *args: updates.append(True))
    scheduler = SimpleNamespace(step=lambda: scheduler_steps.append(True), get_last_lr=lambda: [.01])
    policy = SimpleNamespace(device="cpu", dtype="float32",
                             get_optimizer_preset=lambda: SimpleNamespace(build=lambda _: optimizer, grad_clip_norm=1.),
                             get_scheduler_preset=lambda: SimpleNamespace(build=lambda *_: scheduler))
    wire = {"camera_keys": {}}
    monkeypatch.setattr(training, "_make_data", lambda config: (
        dict(config), policy, None, Data(), Data(), {}, wire, {},
    ))
    monkeypatch.setattr(training, "make_processors", lambda *_: (lambda batch: batch, None))
    monkeypatch.setattr(adapter, "create_pi05_policy", lambda _: model)
    monkeypatch.setattr(training, "_save_checkpoint", lambda output, step, *args, **kwargs: checkpoints.append(step))
    output = tmp_path / "run"
    result = training._train({
        "output_dir": str(output), "steps": 3, "per_device_batch_size": 2,
        "gradient_accumulation_steps": 2, "num_workers": 0, "eval_every": 2,
        "save_every": 3, "log_every": 1,
    }, rank=0, world_size=1, local_device=None)
    assert result["step"] == len(updates) == len(scheduler_steps) == 3
    assert result["global_batch_size"] == 4
    assert checkpoints == [2, 3]
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [record["global_batch_samples"] for record in records] == [4, 3, 3]
    assert all(record["gradient_accumulation_steps"] == 2 for record in records)
