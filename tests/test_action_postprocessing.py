"""CPU checks for causal filtering and an immutable, strictly validated eval."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from pressb.online_rl.action_postprocessing import (
    CausalActionPostprocessor, SmoothedResidualOnNoiseRunner, quaternion_slerp,
)
from pressb.online_rl.residual_on_noise import ResidualOnNoiseRunner, actor_sha256
from test_fast_online_rl_runner import BatchInference, IndexedSimulation, config
from test_online_rl_runner import TOKEN, serve
from test_residual_on_noise import noise_checkpoint


def state(x=0.):
    return np.array([x, 0., 0., 1., 0., 0., 0., .008])


def chunk(x=1.):
    raw = np.tile(state(x), (7, 1))
    raw[:, -1] = np.linspace(.005, .012, 7)
    return raw


@pytest.mark.parametrize("mode", ["xyz_ema", "pose_ema"])
def test_step_response_matches_closed_form_and_preserves_gripper(mode):
    processor = CausalActionPostprocessor(mode, .6)
    raw = chunk()
    sent, record = processor.apply(0, "first", raw, state())
    np.testing.assert_allclose(sent[:, 0], 1 - .4 ** np.arange(1, 8))
    np.testing.assert_array_equal(sent[:, -1], raw[:, -1])
    np.testing.assert_array_equal(raw, chunk())
    assert record["reset"] is True and record["chunk_index"] == 0
    np.testing.assert_array_equal(record["previous_sent_pose8"], state())


def test_ramp_is_causal_and_future_targets_cannot_change_previous_commands():
    raw = chunk()
    raw[:, 0] = np.arange(7) * .02
    raw2 = raw.copy()
    raw2[4:, 0] = 1000.
    first, _ = CausalActionPostprocessor("xyz_ema", .5).apply(1, "a", raw, state())
    second, _ = CausalActionPostprocessor("xyz_ema", .5).apply(1, "a", raw2, state())
    np.testing.assert_array_equal(first[:4], second[:4])
    expected = [0., .01, .025, .0425, .06125, .080625, .1003125]
    np.testing.assert_allclose(first[:, 0], expected)


def test_state_carries_across_chunks_and_ignores_later_measured_position():
    processor = CausalActionPostprocessor("xyz_ema", .5)
    first, _ = processor.apply(0, "same", chunk(), state())
    second, record = processor.apply(0, "same", chunk(), state(50.))
    np.testing.assert_allclose(np.r_[first[:, 0], second[:, 0]], 1 - .5 ** np.arange(1, 15))
    np.testing.assert_array_equal(record["previous_sent_pose8"], first[-1])
    assert not record["reset"] and record["chunk_index"] == 1


def test_slots_are_isolated_and_episode_change_resets_to_measured_pose():
    processor = CausalActionPostprocessor("xyz_ema", .5)
    first, _ = processor.apply(0, "same", chunk(), state())
    other, _ = processor.apply(1, "other", chunk(12.), state(10.))
    assert other[0, 0] == 11.
    continued, _ = processor.apply(0, "same", chunk(), state())
    assert continued[0, 0] == pytest.approx(.5 + .5 * first[-1, 0])
    reset, metadata = processor.apply(0, "new", chunk(4.), state(2.))
    assert reset[0, 0] == 3.
    assert metadata["reset"] and metadata["chunk_index"] == 0
    assert len(processor._slots) == 2


@pytest.mark.parametrize("mode,alpha", [("none", .2), ("xyz_ema", 1.)])
def test_identity_path_exactly_preserves_raw_numeric_values(mode, alpha):
    raw = chunk(-.25)
    raw[:, 3:7] = np.array([-.70710677, .70710677, 0., 0.], dtype=np.float32)
    sent, _ = CausalActionPostprocessor(mode, alpha).apply(0, "a", raw, state(10.))
    np.testing.assert_array_equal(sent, raw)


def test_xyz_filter_does_not_renormalize_or_recanonicalize_quaternion():
    raw = chunk()
    raw[:, 3:7] = np.array([-.70710677, .70710677, 0., 0.], dtype=np.float32)
    sent, _ = CausalActionPostprocessor("xyz_ema", .6).apply(0, "a", raw, state())
    np.testing.assert_array_equal(sent[:, 3:], raw[:, 3:])


@pytest.mark.parametrize("sign", [1., -1.])
def test_quaternion_slerp_uses_shortest_arc_and_normalizes(sign):
    # +270 degrees is the same orientation as -90 degrees. Halfway must be
    # -45 degrees rather than +135 degrees, regardless of quaternion sign.
    target = sign * np.array([np.cos(3*np.pi/4), 0., 0., np.sin(3*np.pi/4)]) * .99999
    result = quaternion_slerp([1., 0., 0., 0.], target, .5)
    expected = [np.cos(np.pi/8), 0., 0., -np.sin(np.pi/8)]
    np.testing.assert_allclose(result, expected, atol=1e-14)
    assert np.linalg.norm(result) == pytest.approx(1.)


def test_antipodal_and_tiny_angle_quaternions_remain_finite_and_unit():
    q = np.array([.5, .5, .5, .5])
    np.testing.assert_allclose(quaternion_slerp(q, -q, .6), q, atol=1e-15)
    near = quaternion_slerp([1., 0., 0., 0.], [1., 1e-10, 0., 0.], .6)
    assert np.linalg.norm(near) == pytest.approx(1.)
    assert near[1] == pytest.approx(6e-11)
    with pytest.raises(ValueError, match="zero quaternion"):
        quaternion_slerp(np.zeros(4), q, .5)


def test_pose_filter_orientation_state_carries_across_chunks_and_resets():
    raw = chunk()
    raw[:, 3:7] = [0., 0., 0., 1.]
    processor = CausalActionPostprocessor("pose_ema", .5)
    first, _ = processor.apply(0, "a", raw, state())
    second, _ = processor.apply(0, "a", raw, state())
    expected_angles = np.pi * (1 - .5 ** np.arange(1, 15))
    combined = np.concatenate([first, second])
    np.testing.assert_allclose(2*np.arctan2(combined[:, 6], combined[:, 3]), expected_angles,
                               atol=2e-7)
    np.testing.assert_allclose(np.linalg.norm(combined[:, 3:7], axis=1), 1., atol=1e-14)
    reset, _ = processor.apply(0, "b", raw, state())
    np.testing.assert_allclose(reset[0], first[0], atol=1e-14)


def test_xyz_output_stays_in_convex_hull_of_previous_and_current_targets():
    rng = np.random.default_rng(23)
    initial = state()
    initial[:3] = rng.uniform(-1., 1., 3)
    raw = chunk()
    raw[:, :3] = rng.uniform(-1., 1., (7, 3))
    sent, _ = CausalActionPostprocessor("xyz_ema", .6).apply(0, "a", raw, initial)
    previous = initial
    for target, filtered in zip(raw, sent, strict=True):
        assert np.all(filtered[:3] >= np.minimum(previous[:3], target[:3]))
        assert np.all(filtered[:3] <= np.maximum(previous[:3], target[:3]))
        previous = filtered


@pytest.mark.parametrize("alpha", [0., -.2, 1.001, True, "0.6", float("nan"), float("inf")])
def test_invalid_alpha_is_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        CausalActionPostprocessor("xyz_ema", alpha)


@pytest.mark.parametrize("mode,residual_mode", [("train", "xyz"), ("eval", "pose9")])
def test_training_and_non_xyz_configuration_are_rejected_before_side_effects(tmp_path, mode, residual_mode):
    cfg = config(mode=mode, learner={"residual_mode": residual_mode})
    with pytest.raises(ValueError):
        SmoothedResidualOnNoiseRunner(cfg, "http://unused", "http://unused", tmp_path / "forbidden",
            checkpoint=tmp_path / "actor.pt", noise_checkpoint=tmp_path / "noise.pt")
    assert not (tmp_path / "forbidden").exists()


@pytest.fixture(scope="module")
def trained_xyz(tmp_path_factory):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    output = tmp_path_factory.mktemp("smoothed_xyz")
    noise_path = output / "noise.pt"
    cfg = config(max_transitions=3, learning_start=2, pipeline_updates=False)
    cfg.learner.update(residual_mode="xyz", residual_scale=[.02, .02, .02])
    simulation, inference = IndexedSimulation(), BatchInference()
    with serve(simulation.handlers) as sim_url, serve(inference.handlers) as infer_url:
        runner = ResidualOnNoiseRunner(cfg, sim_url, infer_url, output / "train",
            noise_checkpoint=noise_path, device="cpu", token=TOKEN, timeout=5)
        runner._verify_nodes()
        noise_checkpoint(noise_path, obs_dim=runner.obs_dim, identities=deepcopy(runner.identities))
        runner.run()
    yield noise_path, runner.output / "last.pt", cfg, actor_sha256(runner.learner.actor)
    torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("mode", ["none", "xyz_ema", "pose_ema"])
def test_cpu_http_evaluation_sends_logged_actions_and_never_updates(trained_xyz, tmp_path, mode):
    noise_path, checkpoint, cfg, expected_actor = trained_xyz
    cfg = deepcopy(cfg)
    cfg.mode, cfg.eval_episodes = "eval", 5
    simulation, inference = IndexedSimulation(), BatchInference()
    with serve(simulation.handlers) as sim_url, serve(inference.handlers) as infer_url:
        runner = SmoothedResidualOnNoiseRunner(cfg, sim_url, infer_url, tmp_path / mode,
            noise_checkpoint=noise_path, checkpoint=checkpoint, device="cpu", token=TOKEN, timeout=5,
            smoothing_mode=mode, smoothing_alpha=.6)
        summary = runner.run()
    assert summary["state"] == "complete" and summary["episodes"] == 5 and summary["updates"] == 0
    assert actor_sha256(runner.learner.actor) == expected_actor
    assert len(runner.replay) == 0
    payload = torch.load(checkpoint, weights_only=False, map_location="cpu")
    assert runner.identities == payload["extra"]["identities"]
    assert runner.learner.updates == payload["updates"]
    logged = [json.loads(line) for line in (runner.output / "action_postprocessing.jsonl").read_text().splitlines()]
    actual = [action for step in simulation.steps for action in step["actions"]]
    assert len(logged) == len(actual)
    for record, wire in zip(logged, actual, strict=True):
        assert wire["env_id"] == record["env_id"]
        np.testing.assert_array_equal(wire["actions_pose8"], record["sent_actions_pose8"])
        raw, sent = np.array(record["raw_actions_pose8"]), np.array(record["sent_actions_pose8"])
        np.testing.assert_array_equal(raw[:, -1], sent[:, -1])
        if mode == "none":
            np.testing.assert_array_equal(raw, sent)
    assert any(record["chunk_index"] > 0 and not record["reset"] for record in logged)
    assert sum(record["reset"] for record in logged) == 5
    manifest = json.loads((runner.output / "postprocessing.json").read_text())
    assert manifest["freeze_verification_passed"]
    assert manifest["residual_actor_sha256_at_end"] == expected_actor
    assert manifest["inference_time_override"] and manifest["trained_simulation_identity_unchanged"]


@pytest.mark.parametrize("change", ["simulator", "checkpoint_mode"])
def test_eval_keeps_original_identity_and_checkpoint_validation(trained_xyz, tmp_path, change):
    noise_path, checkpoint, cfg, _ = trained_xyz
    cfg = deepcopy(cfg)
    cfg.mode = "eval"
    simulation, inference = IndexedSimulation(), BatchInference()
    if change == "simulator":
        simulation.image_resolution = [640, 480]
    else:
        payload = torch.load(checkpoint, weights_only=False, map_location="cpu")
        payload["config"]["residual_mode"] = "pose9"
        payload["config"]["residual_scale"] = [.02]*9
        checkpoint = tmp_path / "wrong_mode.pt"
        torch.save(payload, checkpoint)
    with serve(simulation.handlers) as sim_url, serve(inference.handlers) as infer_url:
        runner = SmoothedResidualOnNoiseRunner(cfg, sim_url, infer_url, tmp_path / "rejected",
            noise_checkpoint=noise_path, checkpoint=checkpoint, device="cpu", token=TOKEN, timeout=5,
            smoothing_mode="xyz_ema", smoothing_alpha=.6)
        with pytest.raises((ValueError, RuntimeError)):
            runner.run()
    assert not simulation.resets and not inference.encoded
