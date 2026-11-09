"""Dataset timestamp alignment, train-only normalization, and SFT integration."""
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")
av = pytest.importorskip("av")

from pressb.training.diffusion_policy_data import LeRobotData, DiffusionDataset


def source(tmp_path):
    root = tmp_path / "source"
    for folder in ("meta/episodes/chunk-000", "data/chunk-000", "videos/observation.images.up/chunk-000"):
        (root / folder).mkdir(parents=True)
    info = dict(codebase_version="v3.0", fps=30, total_frames=8, total_episodes=2,
                video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    (root / "meta/info.json").write_text(json.dumps(info))
    rows = [dict(index=i, episode_index=i//4, **{"observation.state": [i * 10., i], "action": [i*20., i+1.]}) for i in range(8)]
    pq.write_table(pa.Table.from_pylist(rows), root / "data/chunk-000/file-000.parquet")
    episodes = [dict(episode_index=i, dataset_from_index=i*4, dataset_to_index=(i+1)*4,
                     length=4, tasks=["Press the red button."], **{
                         "videos/observation.images.up/chunk_index": 0,
                         "videos/observation.images.up/file_index": 0,
                         "videos/observation.images.up/from_timestamp": i*4/30+.03}) for i in range(2)]
    pq.write_table(pa.Table.from_pylist(episodes), root / "meta/episodes/chunk-000/file-000.parquet")
    video = root / "videos/observation.images.up/chunk-000/file-000.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        for i in range(8):
            image = np.full((64,64,3), 20+i*25, np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return dict(root=str(root), cameras=["up"], resolution=64,
                state_to_si_scale=[np.pi/180, .01], action_to_si_scale=[np.pi/180, .01],
                video_timestamp_mode="packed_validated", metadata_timestamp_offset_s=.03)


def test_packed_video_alignment_units_tails_split_and_normalizer(tmp_path):
    data = LeRobotData(source(tmp_path))
    data.prepare_images(tmp_path / "cache")
    assert all(abs(data.images[0][i].mean() - (20+i*25)) < 6 for i in range(8))
    np.testing.assert_allclose(data.states[1], [np.pi/18, .01])
    np.testing.assert_allclose(data.actions[1], [np.pi/9, .02])
    split = data.split(4)
    assert set(split["train"]).isdisjoint(split["val"])
    stats = data.normalization([0])
    assert stats["state_center"][0] == pytest.approx(np.pi/12)
    sample = DiffusionDataset(data, [0], 8, image_transforms={"enable": False})[3]
    np.testing.assert_allclose(sample["actions"], np.repeat(data.actions[3:4], 8, axis=0))
    # Train identity does not depend on where a source tree/cache is copied.
    data.prepare_images(tmp_path / "cache")


def test_validation_never_augments_and_training_keyed_epoch(tmp_path):
    data = LeRobotData(source(tmp_path))
    data.prepare_images(tmp_path / "cache")
    config = dict(enable=True, max_num_transforms=1, tfs={"brightness": dict(type="ColorJitter",weight=1.,kwargs={"brightness":[.2,.6]})})
    train = DiffusionDataset(data, [0], 8, image_transforms=config)
    val = DiffusionDataset(data, [0], 8, image_transforms=config, split="val")
    raw = data.images[0][2].copy()
    a = train[2]["images"]
    assert torch.equal(a, train[2]["images"])
    train.set_epoch(3)
    assert not torch.equal(a, train[2]["images"])
    assert np.array_equal(raw, val[2]["images"][0].permute(1,2,0))
    assert np.array_equal(raw, data.images[0][2])


def test_reject_wrong_packed_offset_and_cache_identity(tmp_path):
    config = source(tmp_path)
    with pytest.raises(ValueError, match="offset"):
        LeRobotData({**config, "metadata_timestamp_offset_s": 0})
    data = LeRobotData(config)
    data.prepare_images(tmp_path / "cache")
    other = LeRobotData({**config, "resolution": 32})
    with pytest.raises(ValueError, match="identity"):
        other.prepare_images(tmp_path / "cache")


def test_full_cpu_sft_and_reload(tmp_path):
    from pressb.training.diffusion_policy import train, load_checkpoint
    config = dict(seed=2, device="cpu", cpu_threads=1, output_dir=str(tmp_path / "run"),
        image_cache_dir=str(tmp_path / "cache"), num_workers=0, batch_size=2, max_steps=2,
        eval_every=1, validation_samples=4, warmup_steps=1, image_transforms={"enable":False},
        dataset=source(tmp_path), model=dict(horizon=8, diffusion_steps=10, inference_steps=5,
        unet_dim=8, unet_dim_mults=[1,2], time_embedding_dim=8, keypoints=2,task_embedding_dim=2,
        state_names=["joint1","gripper_width"], state_units=["rad","m"],
        action_names=["joint1","gripper_width"], action_units=["rad","m"]))
    result = train(config)
    assert result["step"] == 2 and np.isfinite(result["best_validation_loss"])
    model, saved = load_checkpoint(tmp_path / "run/last.pt")
    assert saved["step"] == 2 and model.horizon == 8
    assert saved["training_contract"]["split"]["train"] != saved["training_contract"]["split"]["val"]
    assert (tmp_path / "run/best.pt").is_file()


def test_resume_reproduces_optimizer_suffix_and_rejects_changed_contract(tmp_path, monkeypatch):
    import pressb.training.diffusion_policy as trainer
    config = dict(seed=2, device="cpu", cpu_threads=1, output_dir=str(tmp_path / "full"),
        image_cache_dir=str(tmp_path / "cache"), num_workers=0, batch_size=2, max_steps=2,
        eval_every=1, validation_samples=4, warmup_steps=1, image_transforms={"enable":False},
        dataset=source(tmp_path), model=dict(horizon=8, diffusion_steps=10, inference_steps=5,
        unet_dim=8, unet_dim_mults=[1,2], time_embedding_dim=8, keypoints=2,task_embedding_dim=2,
        state_names=["joint1","gripper_width"], state_units=["rad","m"],
        action_names=["joint1","gripper_width"], action_units=["rad","m"]))
    trainer.train(config)
    interrupted = {**config, "output_dir": str(tmp_path / "interrupted")}
    original_save = trainer.save_checkpoint

    class Interrupted(Exception):
        pass

    def save_then_interrupt(path, payload):
        original_save(path, payload)
        if Path(path).name == "last.pt" and payload["step"] == 1:
            raise Interrupted()

    monkeypatch.setattr(trainer, "save_checkpoint", save_then_interrupt)
    with pytest.raises(Interrupted):
        trainer.train(interrupted)
    monkeypatch.setattr(trainer, "save_checkpoint", original_save)
    resumed = {**interrupted, "resume": str(tmp_path / "interrupted/last.pt")}
    trainer.train(resumed)
    full = torch.load(tmp_path / "full/last.pt", weights_only=True)
    again = torch.load(tmp_path / "interrupted/last.pt", weights_only=True)
    for kind in ("model", "ema"):
        assert all(torch.equal(value, again[kind][name]) for name, value in full[kind].items())
    before = (tmp_path / "interrupted/resolved_config.json").read_bytes()
    with pytest.raises(ValueError, match="contract differs"):
        trainer.train({**resumed, "batch_size": 1})
    assert (tmp_path / "interrupted/resolved_config.json").read_bytes() == before
