"""Small CPU checks for checkpoint honesty, validation and DDP loss weighting."""
import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location("piper_training", Path(__file__).parents[1] / "starVLA/training/train_piper.py")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


def checkpoint():
    source = {prefix + "weight": torch.ones(2, 3) for prefix in trainer.COMPONENTS[:3]}
    target = dict(source)
    for name, src, dst in (
        ("action_encoder.layer1.weight", (3, 7), (3, 9)),
        ("state_encoder.layer1.weight", (5, 8), (5, 9)),
        ("action_decoder.layer2.weight", (7, 5), (9, 5)),
        ("action_decoder.layer2.bias", (7,), (9,)),
    ):
        source["action_model." + name] = torch.ones(src)
        target["action_model." + name] = torch.ones(dst)
    source["action_model.kept.weight"] = target["action_model.kept.weight"] = torch.ones(4, 4)
    return source, target


def test_only_declared_interface_shapes_are_adapted():
    source, target = checkpoint()
    loaded, report = trainer.select_pretrained_state(source, target)
    assert len(report["adapted_keys"]) == 4
    assert "action_model.kept.weight" in loaded
    assert report["head_initialization"] == "pretrained_head_with_new_9d_interfaces"


def test_missing_whole_head_is_explicit_and_rejected():
    source, target = checkpoint()
    source = {k: v for k, v in source.items() if not k.startswith("action_model.")}
    with pytest.raises(ValueError, match="Entire pretrained action head is absent"):
        trainer.select_pretrained_state(source, target)
    source, target = checkpoint()
    del source["vj_predictor.weight"]
    with pytest.raises(ValueError, match="Missing pretrained"):
        trainer.select_pretrained_state(source, target)


@pytest.mark.parametrize("key,shape", [("vj_predictor.weight", (3, 3)),
                                      ("action_model.action_encoder.layer1.weight", (100, 7)),
                                      ("action_model.state_encoder.layer1.weight", (5, 1))])
def test_wrong_checkpoint_is_rejected(key, shape):
    source, target = checkpoint()
    source[key] = torch.ones(shape)
    with pytest.raises(ValueError):
        trainer.select_pretrained_state(source, target)


def test_partial_missing_head_and_unexpected_keys_rejected():
    source, target = checkpoint()
    source.pop("action_model.kept.weight")
    with pytest.raises(ValueError, match="Missing pretrained"):
        trainer.select_pretrained_state(source, target)
    source, target = checkpoint()
    source["surprise.weight"] = torch.ones(1)
    with pytest.raises(ValueError, match="Unexpected"):
        trainer.select_pretrained_state(source, target)


def test_row_major_projection_angle_and_degenerate_penalty():
    identity = np.array([[1., 0, 0, 0, 1, 0]])
    rz90 = np.array([[0., -1, 0, 1, 0, 0]])
    error, invalid = trainer.rotation_errors_deg(rz90 * 3, identity)
    assert error[0] == pytest.approx(90.)
    assert not invalid[0]
    error, invalid = trainer.rotation_errors_deg(identity * 0, identity)
    assert error[0] == 180 and invalid[0]


def test_validation_anchors_are_even_and_episode_boundaries_preserved():
    class Data:
        source_episode_ids = [2, 99]
        episode_lengths = {2: 11, 99: 3}
    anchors = trainer.validation_anchors(Data(), 5)
    assert anchors == [(2, 0), (2, 2), (2, 5), (2, 8), (2, 10), (99, 0), (99, 1), (99, 2)]
    assert len(trainer.validation_anchors(Data(), 5, 4)) == 4


def test_ddp_weighted_accumulation_matches_global_valid_element_mean():
    def example(actions, frames):
        return dict(action_is_pad=np.arange(7) >= actions, video_is_pad=np.arange(8) >= frames)
    batches = [[example(7, 8)], [example(1, 1)]]
    # Other rank has 2 and 3 valid actions; 1 and 2 future tubelets.
    with patch.object(trainer.dist, "all_reduce", side_effect=lambda t: t.add_(torch.tensor([5., 3.]))):
        weights = trainer.loss_weights(batches, 2, torch.device("cpu"), 2)
    assert torch.allclose(weights[:, 0], torch.tensor([14/13, 2/13]))
    assert torch.allclose(weights[:, 1], torch.tensor([1., 0.]))


def test_resume_contract_rejects_changed_sampling_and_schedule():
    from omegaconf import OmegaConf
    path = Path(__file__).parents[1] / "scripts/configs/vlajepa_piper_ft.yaml"
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    _, before = trainer.training_contract(cfg)
    cfg["run_id"] = "another_log_path"
    cfg["framework"]["qwenvl"]["device_map"] = "cuda:3"
    assert trainer.training_contract(cfg)[1] == before
    cfg["datasets"]["vla_data"]["per_device_batch_size"] += 1
    assert trainer.training_contract(cfg)[1] != before
    cfg["datasets"]["vla_data"]["per_device_batch_size"] -= 1
    cfg["trainer"]["max_train_steps"] += 1
    assert trainer.training_contract(cfg)[1] != before


def tied_model():
    model = torch.nn.Module()
    model.embedding = torch.nn.Embedding(3, 2)
    model.head = torch.nn.Linear(2, 3, bias=False)
    model.head.weight = model.embedding.weight
    return model


def test_equal_separate_source_storages_are_safe_for_tied_target():
    model = tied_model()
    source = {name: value.clone() for name, value in model.state_dict().items()}
    assert source["embedding.weight"].data_ptr() != source["head.weight"].data_ptr()
    assert trainer.validate_tied_source(model, source) == [["embedding.weight", "head.weight"]]


def test_conflicting_tied_source_is_rejected_before_overwriting_target():
    model = tied_model()
    original = model.embedding.weight.detach().clone()
    source = {name: value.clone() for name, value in model.state_dict().items()}
    source["head.weight"][0, 0] += 1
    with pytest.raises(ValueError, match="Conflicting source tensors for tied target"):
        trainer.validate_tied_source(model, source)
    assert torch.equal(model.embedding.weight, original)


def test_partial_tied_alias_adaptation_is_rejected():
    model = tied_model()
    with pytest.raises(ValueError, match="Only some aliases"):
        trainer.validate_tied_source(model, {"embedding.weight": model.embedding.weight.detach().clone()})


def test_three_epoch_geometry_keeps_partial_batches_and_normalizes_final_checkpoint():
    geometry = trainer.epoch_geometry(275687, 4, 16, 1, 3)
    assert geometry["samples_per_rank"] == 68922
    assert geometry["batches_per_epoch"] == geometry["steps_per_epoch"] == 4308
    assert geometry["target_steps"] == 12924
    assert geometry["sampler_padding_per_epoch"] == 1
    progress = dict(step=2000, epoch=0, batch_in_epoch=2000)
    while progress["step"] < geometry["target_steps"]:
        progress["step"] += 1
        progress["batch_in_epoch"] += 1
        trainer.normalize_epoch_cursor(progress, geometry["batches_per_epoch"])
    assert progress == dict(step=12924, epoch=3, batch_in_epoch=0)
    assert trainer.epoch_geometry(101, 4, 4, 2, 3)["target_steps"] == 12
    with pytest.raises(ValueError, match="positive integer"):
        trainer.epoch_geometry(101, 4, 4, 2, 0)


def test_offset_sampler_preserves_real_four_rank_suffix_without_decoding_prefix():
    from torch.utils.data import DataLoader, DistributedSampler
    class CountingDataset:
        def __init__(self):
            self.reads = []
        def __len__(self):
            return 275687
        def __getitem__(self, index):
            self.reads.append(index)
            return index
    for rank in range(4):
        data = CountingDataset()
        reference = DistributedSampler(data, num_replicas=4, rank=rank, shuffle=True, seed=42)
        sampler = trainer.OffsetDistributedSampler(data, num_replicas=4, rank=rank, shuffle=True, seed=42)
        expected = list(reference)
        sampler.set_start_index(2000 * 16)
        loader = DataLoader(data, batch_size=16, sampler=sampler)
        consumed = [index for batch in loader for index in batch.tolist()]
        assert consumed == expected[32000:]
        assert data.reads == consumed
        assert len(loader) == 2308
        assert len(consumed) % 16 == 10
        sampler.set_epoch(1)
        sampler.set_start_index(0)
        reference.set_epoch(1)
        assert list(sampler) == list(reference)
        assert len(loader) == 4308


def continuation_fixture():
    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(Path(__file__).parents[1] / "scripts/configs/vlajepa_piper_ft.yaml"), resolve=True)
    learning_rates = cfg["trainer"]["learning_rate"]
    training = dict(step=2000, scheduler=dict(last_epoch=2000), optimizer=dict(param_groups=[
        dict(name=name, initial_lr=learning_rates[name], lr=.1 * learning_rates[name])
        for name in ("qwen_vl_interface", "vj_predictor", "action_model")]))
    return cfg, training


def test_continuation_only_allows_horizon_and_explicit_schedule_changes():
    import copy
    cfg, state = continuation_fixture()
    original, original_fingerprint = trainer.training_contract(cfg)
    cfg["trainer"].update(max_train_steps=12924, target_epochs=3,
                          continuation_schedule=trainer.continuation_schedule(state, cfg["trainer"]["learning_rate"], 12924, 100))
    current, current_fingerprint = trainer.training_contract(cfg)
    assert current_fingerprint != original_fingerprint
    trainer.validate_continuation_contract(original, current)
    for path, key, value in (("data", "per_device_batch_size", 8),
                             ("data", "sample_stride", 2),
                             ("trainer", "eval_interval", 400),
                             ("trainer", "weight_decay", .2)):
        changed = copy.deepcopy(current)
        changed[path][key] = value
        with pytest.raises(ValueError, match="semantics"):
            trainer.validate_continuation_contract(original, changed)
    changed = copy.deepcopy(current)
    changed["framework"]["action_model"]["state_dim"] = 8
    with pytest.raises(ValueError, match="semantics"):
        trainer.validate_continuation_contract(original, changed)


def test_continuation_lr_preserves_checkpoint_rate_then_warms_and_decays():
    cfg, state = continuation_fixture()
    schedule = trainer.continuation_schedule(state, cfg["trainer"]["learning_rate"], 12924, 100)
    functions = trainer.lr_functions(12924, 100, schedule)
    for factor in functions:
        assert factor(2000) == pytest.approx(.1)
        assert factor(2001) == pytest.approx(.109)
        assert factor(2050) == pytest.approx(.55)
        assert factor(2100) == pytest.approx(1.)
        assert factor(7512) == pytest.approx(.55)
        assert factor(12924) == pytest.approx(.1)
    state["scheduler"]["last_epoch"] = 1999
    with pytest.raises(ValueError, match="scheduler cursor"):
        trainer.continuation_schedule(state, cfg["trainer"]["learning_rate"], 12924, 100)


def test_adamw_moments_and_segment_schedule_survive_strict_resume(tmp_path):
    """A saved continuation resumes to the same next update and LR as uninterrupted."""
    def optimizer_for(parameters):
        names = ("qwen_vl_interface", "vj_predictor", "action_model")
        return torch.optim.AdamW([dict(params=[parameter], name=name, lr=.01)
                                 for parameter, name in zip(parameters, names)], betas=(.9, .95), foreach=False)
    def update(parameters, optimizer, scheduler):
        sum((parameter ** 2).sum() for parameter in parameters).backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    params = [torch.nn.Parameter(torch.tensor([1., 2.])) for _ in range(3)]
    optimizer = optimizer_for(params)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, trainer.lr_functions(8, 2))
    for _ in range(8):
        update(params, optimizer, scheduler)
    state = dict(step=8, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
    schedule = trainer.continuation_schedule(state, {name: .01 for name in ("qwen_vl_interface", "vj_predictor", "action_model")}, 30, 3)
    moments = [optimizer.state[parameter]["exp_avg"].clone() for parameter in params]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, trainer.lr_functions(30, 3, schedule), last_epoch=7)
    assert scheduler.last_epoch == 8
    assert scheduler.get_last_lr() == pytest.approx([.001] * 3)
    for parameter, moment in zip(params, moments):
        assert torch.equal(optimizer.state[parameter]["exp_avg"], moment)
    for _ in range(4):
        update(params, optimizer, scheduler)
    checkpoint = tmp_path / "training.pt"
    torch.save(dict(parameters=[parameter.detach() for parameter in params], optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()), checkpoint)
    saved = torch.load(checkpoint, weights_only=False)
    resumed_params = [torch.nn.Parameter(value.clone()) for value in saved["parameters"]]
    resumed_optimizer = optimizer_for(resumed_params)
    resumed_optimizer.load_state_dict(saved["optimizer"])
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, trainer.lr_functions(30, 3, schedule))
    resumed_scheduler.load_state_dict(saved["scheduler"])
    for group, rate in zip(resumed_optimizer.param_groups, resumed_scheduler.get_last_lr()):
        group["lr"] = rate
    update(params, optimizer, scheduler)
    update(resumed_params, resumed_optimizer, resumed_scheduler)
    assert resumed_scheduler.last_epoch == scheduler.last_epoch == 13
    assert resumed_scheduler.get_last_lr() == scheduler.get_last_lr()
    for original, resumed in zip(params, resumed_params):
        assert torch.equal(original, resumed)
        assert torch.equal(optimizer.state[original]["exp_avg"], resumed_optimizer.state[resumed]["exp_avg"])
