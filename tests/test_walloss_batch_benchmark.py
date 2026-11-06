"""Benchmark validation without model weights, network or CUDA allocation."""
import argparse
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("walloss_batch_benchmark", ROOT / "scripts/benchmark_walloss_batch.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_bounded_batch_sizes():
    assert benchmark.parse_batch_sizes("1,4,64") == [1, 4, 64]
    for invalid in ("", "0", "65", "1,1", "a"):
        with pytest.raises(argparse.ArgumentTypeError):
            benchmark.parse_batch_sizes(invalid)


def test_reports_error_in_separate_physical_units():
    reference = np.zeros((2, 32, 7))
    actual = reference.copy()
    actual[..., :6] = .01
    actual[..., 6] = .002
    errors = benchmark.action_errors(actual, reference)
    assert errors["arm_rad"]["max_abs"] == pytest.approx(.01)
    assert errors["gripper_m"]["rms"] == pytest.approx(.002)
    with pytest.raises(ValueError, match="Nonfinite"):
        benchmark.error_stats([float("nan")], [0.])


def test_observation_sources_are_preserved_and_prompt_lengths_vary(tmp_path):
    source = {"state": [1.] * 7, "images": {"up": {}}, "task": "Move it."}
    path = tmp_path / "observations.json"
    path.write_text(json.dumps([source, {**source, "state": [2.] * 7}]))
    sources, identities = benchmark.load_observations([path])
    outputs = benchmark.make_observations(sources, 8, True)
    assert identities[0]["observations"] == 2
    assert len({len(item["task"]) for item in outputs}) == 4
    assert outputs[1]["state"][0] == 2.
    outputs[0]["state"][0] = 99
    assert sources[0]["state"][0] == 1.


def test_hooks_cannot_mistake_serial_calls_for_real_batch():
    encode = {"vision_calls": 8, "vision_image_count_histogram": {"2": 8}}
    decode = {"vision_calls": 0, "transformer_batch_histogram": {"1": 80}}
    result = benchmark.prove_batch(encode, decode, 8, 2)
    assert not result["single_native_batch"]
    assert not result["not_serial_single_environment_calls"]
    encode.update(vision_calls=1, vision_image_count_histogram={"16": 1})
    decode["transformer_batch_histogram"] = {"8": 10}
    result = benchmark.prove_batch(encode, decode, 8, 2)
    assert result["single_native_batch"] and result["decode_reuses_vision_cache"]


def fake_backend(torch, mutate=False):
    class Module(torch.nn.Module):
        def forward(self, hidden_states=None, grid_thw=None, inputs_embeds=None):
            return hidden_states if hidden_states is not None else inputs_embeds

    class Policy:
        def __init__(self):
            self.model = SimpleNamespace(visual=Module(), model=Module())
            self.model.config = SimpleNamespace(_attn_implementation="eager")
            self.model.visual.config = SimpleNamespace(_attn_implementation="sdpa")

        def encode(self, observation, num_inference_timesteps=10):
            return self.encode_batch([observation], num_inference_timesteps)[0]

        def encode_batch(self, observations, num_inference_timesteps=10):
            self.model.visual(torch.ones((len(observations) * 4, 2)), grid_thw=torch.ones((len(observations) * 2, 3)))
            return [{"feature": torch.tensor([[float(len(item["task"]))]]), "inputs": {
                "inputs_embeds": torch.tensor([[[float(len(item["task"]))]]]),
                "attention_mask": torch.ones((1, len(item["task"])), dtype=torch.long)}} for item in observations]

        def decode(self, context, initial_noise):
            return self.decode_batch([context], initial_noise)

        def decode_batch(self, contexts, initial_noise):
            embeddings = torch.cat([item["inputs"]["inputs_embeds"] for item in contexts])
            self.model.model(inputs_embeds=embeddings)
            self.model.model(inputs_embeds=embeddings)
            output = initial_noise.numpy() + embeddings.numpy().reshape(-1, 1, 1) * .01
            return {"action": output.astype(np.float32)}

    class Backend:
        config = {"device": "cpu"}
        horizon, noise_dim, num_inference_timesteps = 2, 7, 2
        action_scale = np.array([.1] * 6 + [.01], dtype=np.float32)
        camera_mapping = {"up": "face", "wrist": "wrist"}
        closed = False

        def __init__(self):
            self.policy = Policy()

        def describe(self):
            return {"action": {"units": ["rad"] * 6 + ["m"]}}

        def _prepare_observation(self, value):
            return value

        def encode_batch(self, observations, seeds):
            return [(SimpleNamespace(conditioning=item, baseline_noise=torch.zeros((1, 2, 7))), item["feature"][0].numpy())
                    for item in self.policy.encode_batch(observations)]

        def decode_batch(self, contexts, noises):
            if mutate:
                contexts[0].conditioning["inputs"]["inputs_embeds"] += 1.
            return self.policy.decode_batch([item.conditioning for item in contexts], torch.as_tensor(np.asarray(noises)))["action"] * self.action_scale

        def close(self):
            self.closed = True

    return Backend()


def test_incremental_report_checks_real_hooks_native_reference_and_cache(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from pressb.runtime import server
    backend = fake_backend(torch)
    monkeypatch.setattr(server, "load_backend", lambda config: backend)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"backend": {"factory": "fake:backend"}}))
    observation = tmp_path / "observation.json"
    observation.write_text(json.dumps({"state": [0.] * 7, "images": {}, "task": "Place object."}))
    output = tmp_path / "report.json"
    args = benchmark.parser().parse_args(["--config", str(config), "--observation", str(observation),
        "--output", str(output), "--batch-sizes", "1,4", "--repeats", "2", "--warmups", "0",
        "--max-arm-error-rad", "0.000001", "--max-gripper-error-m", "0.000001"])
    assert benchmark.run(args) == 0
    report = json.loads(output.read_text())
    assert report["state"] == "completed" and backend.closed
    assert report["native_batch_proof_passed"]
    assert report["reference_completed"] == 4
    row = report["results"][1]
    assert row["encode_forward_evidence"]["vision_calls"] == 1
    assert row["decode_forward_evidence"]["transformer_batch_histogram"] == {"4": 2}
    assert row["comparison_vs_native_single"]["actions"]["arm_rad"]["max_abs"] == 0.
    assert row["cache_checks"]["subset"]["indices"] == [3, 0, 2]
    assert row["cache_checks"]["duplicate"]["indices"] == [0, 0]
    assert len(set(row["valid_token_lengths"])) > 1


def test_cache_mutation_is_detected():
    torch = pytest.importorskip("torch")
    backend = fake_backend(torch, mutate=True)
    contexts = [row[0] for row in backend.encode_batch([{"task": "a"}, {"task": "bb"}], [1, 2])]
    noises = np.zeros((2, 2, 7), dtype=np.float32)
    args = SimpleNamespace(max_arm_error_rad=None, max_gripper_error_m=None)
    with pytest.raises(AssertionError, match="mutated cached"):
        benchmark.check_cache(backend, contexts, noises, np.zeros((2, 2, 7)), args)


def test_acceptance_thresholds_are_explicit():
    args = SimpleNamespace(max_feature_error=None, max_arm_error_rad=.1, max_gripper_error_m=.01)
    comparison = benchmark.compare_outputs([[0.]], np.ones((1, 2, 7)), [[0.]], np.zeros((1, 2, 7)), args)
    assert comparison["status"] == "failed"
    assert not comparison["thresholds"]["gripper_m"]["passed"]
