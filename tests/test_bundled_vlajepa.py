"""Bundled-source provenance and real, small flow-head optimization checks."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "src/pressb/model_sources/vlajepa"

# Only these imported files differ from the preserved upstream snapshot.
SFT_SOURCE_ADAPTATIONS = {
    "scripts/configs/vlajepa_piper_ft.yaml",
    "starVLA/dataloader/piper_lerobot.py",
    "starVLA/dataloader/table_buttons_lerobot.py",
    "starVLA/dataloader/table_buttons_multitask_lerobot.py",
    "starVLA/training/train_piper.py",
    "starVLA/training/train_table_buttons.py",
    "starVLA/training/train_table_buttons_multitask.py",
}
SFT_TEST_ADAPTATIONS = {
    "tests/test_piper_trainer.py",
}
FIXTURE_PATH_ADAPTATIONS = {
    "tests/test_table_buttons_1000_training.py",
    "tests/test_table_buttons_multitask_joint6_training.py",
}


def test_bundled_vlajepa_source_inventory():
    manifest = json.loads((SOURCE / "source_manifest.json").read_text())
    assert manifest["commit"] == "9c9dc71c8e199c7428b25d103ab666395832c001"
    assert set(manifest["profiles"]) == {
        "pose9", "joint6", "joint7", "joint6_multitask", "joint7_multitask", "joint7_generic"}
    changed_sources = set()
    for relative, item in manifest["files"].items():
        path = SOURCE / relative
        assert path.is_file(), relative
        assert path.stat().st_size == item["bytes"], relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], relative
        assert path.suffix not in (".pt", ".pth", ".so", ".safetensors")
        if item["source_sha256"] is not None and item["sha256"] != item["source_sha256"]:
            changed_sources.add(relative)
        if relative in SFT_SOURCE_ADAPTATIONS | SFT_TEST_ADAPTATIONS:
            assert item["adaptation"], relative
    assert changed_sources == SFT_SOURCE_ADAPTATIONS | SFT_TEST_ADAPTATIONS | FIXTURE_PATH_ADAPTATIONS
    for relative in (
        "starVLA/model/framework/VLA_JEPA.py",
        "starVLA/model/modules/vlm/QWen3.py",
        "starVLA/model/modules/action_model/GR00T_ActionHeader.py",
        "starVLA/inference/piper_policy.py",
        "starVLA/training/train_table_buttons_multitask_joint6.py",
    ):
        item = manifest["files"][relative]
        assert item["sha256"] == item["source_sha256"], relative


@pytest.mark.parametrize("dimension,state_dimension", [(9, 9), (6, 7), (7, 7)])
def test_bundled_flow_head_real_optimizer_step_and_checkpoint(dimension, state_dimension, tmp_path):
    # Model packages live in an isolated environment. Main simulator/CPU tests
    # can still audit the complete source inventory without installing them.
    for dependency in ("torch", "transformers", "diffusers", "omegaconf"):
        if importlib.util.find_spec(dependency) is None:
            pytest.skip(f"Run in the VLA-JEPA environment: missing {dependency}")
    code = r'''
import io, json, sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
from starVLA.model.modules.action_model import GR00T_ActionHeader as native
torch.set_num_threads(1)
torch.manual_seed(42)
dimension, state_dimension = map(int, sys.argv[1:3])
# Only dimensions are reduced; use the actual unmodified DiT, attention,
# action/state encoders, flow loss, optimizer and checkpoint tensor loader.
native.DiTConfig['test_cpu'] = dict(input_embedding_dim=16, attention_head_dim=8, num_attention_heads=2)
config = OmegaConf.create(dict(framework=dict(action_model=dict(
    action_model_type='test_cpu', hidden_size=16, action_dim=dimension,
    state_dim=state_dimension, action_horizon=7, future_action_window_size=6,
    num_inference_timesteps=2, num_target_vision_tokens=4, add_pos_embed=True,
    max_seq_len=32, noise_beta_alpha=1.5, noise_beta_beta=1., noise_s=.999,
    num_timestep_buckets=1000, diffusion_model_cfg=dict(cross_attention_dim=12,
    num_layers=2, output_dim=16, dropout=0., final_dropout=False,
    interleave_self_attention=True, norm_type='ada_norm', positional_embeddings=None)))))
model = native.FlowmatchingActionHead(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
features = torch.randn(2, 4, 12)
actions = torch.randn(2, 7, dimension)
states = torch.randn(2, 1, state_dimension)
mask = torch.tensor([[False]*7, [False]*4+[True]*3])
before = model.action_decoder.layer2.weight.detach().clone()
loss = model(features, actions, states, action_is_pad=mask)
assert loss.ndim == 0 and torch.isfinite(loss)
loss.backward()
assert model.state_encoder.layer1.weight.grad.abs().sum() > 0
optimizer.step()
assert not torch.equal(before, model.action_decoder.layer2.weight)
stream = io.BytesIO()
torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), stream)
stream.seek(0)
restored = native.FlowmatchingActionHead(config)
checkpoint = torch.load(stream, map_location='cpu', weights_only=True)
restored.load_state_dict(checkpoint['model'], strict=True)
optimizer2 = torch.optim.AdamW(restored.parameters(), lr=1e-3)
optimizer2.load_state_dict(checkpoint['optimizer'])
assert len(optimizer2.state) == len(optimizer.state) > 0
model.eval(); restored.eval()
with torch.no_grad():
    torch.manual_seed(24)
    expected = model.predict_action(features, states)
    torch.manual_seed(24)
    actual = restored.predict_action(features, states)
assert tuple(actual.shape) == (2, 7, dimension)
assert torch.isfinite(actual).all() and torch.equal(actual, expected)
assert Path(sys.modules[native.__name__].__file__).is_relative_to(Path(sys.argv[3]))
print(json.dumps(dict(action_dim=dimension, state_dim=state_dimension, loss=float(loss),
    optimizer_updated=True, restored_prediction_equal=True)))
'''
    env = dict(os.environ, PYTHONPATH=str(SOURCE), CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-c", code, str(dimension), str(state_dimension), str(SOURCE)],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads(result.stdout.splitlines()[-1])
    assert evidence["action_dim"] == dimension
    assert evidence["optimizer_updated"] and evidence["restored_prediction_equal"]
