"""Multi-task metadata selects all three instructions without widening native actions."""
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from PIL import Image

from starVLA.dataloader.table_buttons_joint6_lerobot import NORMALIZATION
from starVLA.inference.table_buttons_multitask_joint6_policy import (
    MODEL_GROUPS, NATIVE_SCHEMA, load_metadata, TableButtonsMultitaskJoint6Policy,
)


def fixture(root, model_id):
    best = root / "checkpoints/best"
    best.mkdir(parents=True)
    (best / "model.pt").write_bytes(b"metadata fixture")
    group = MODEL_GROUPS[model_id]
    provenance = dict(dataset_sha256="dataset", split_sha256="split", training_fingerprint="training")
    config = dict(run_id=model_id,
        framework=dict(action_model=dict(state_dim=7, action_dim=6, action_horizon=7, num_inference_timesteps=4)),
        datasets=dict(vla_data=dict(resolution_size=224, data_root_dir=f"/not-mounted/{group}")))
    OmegaConf.save(OmegaConf.create(config), root / "config.yaml")
    files = {
        "normalization.json": NORMALIZATION,
        "action_schema.json": dict(schema=NATIVE_SCHEMA, state_dim=7, action_dim=6, action_horizon=7,
            cameras=["up", "wrist"], fps=30, normalization=NORMALIZATION,
            fixed_gripper_full_width_m=NORMALIZATION["fixed_gripper_full_width_m"], dataset_group=group, **provenance),
        "checkpoints/best/manifest.json": dict(schema=NATIVE_SCHEMA, complete=True, step=300,
            model_bytes=(best / "model.pt").stat().st_size, **provenance),
        "training_contract.json": dict(contract=dict(source_episode_count=3000,
            split_counts=dict(train=2700, val=300), layout_group_key="layout_group_id", multitask=True)),
    }
    for name, value in files.items():
        (root / name).write_text(json.dumps(value))
    return root


@pytest.mark.parametrize("model_id", list(MODEL_GROUPS))
def test_all_three_profiles_accept_all_colors_and_retain_native_six_actions(tmp_path, model_id):
    run = fixture(tmp_path, model_id)
    _, identity = load_metadata(run)
    assert identity["source_episode_count"] == 3000 and identity["task_type"] == "multitask"
    assert identity["source_dataset_group"] == MODEL_GROUPS[model_id]
    assert identity["native_action_dim"] == 6 and identity["wire_action_dim"] == 7
    assert identity["gripper_learned"] is False
    assert identity["allowed_instructions"] == [f"Press the {color} button." for color in ("red", "yellow", "blue")]
    policy = object.__new__(TableButtonsMultitaskJoint6Policy)
    policy.identity, policy.resolution = identity, 224
    samples = [dict(episode_id=color, request_id=0, instruction=f"Press the {color} button.",
                    state=[0] * 6 + [.021], images=[Image.new("RGB", (640, 480))] * 2, noise_seed=42)
               for color in ("red", "yellow", "blue")]
    images, states, instructions, seeds = policy.prepare(samples)
    assert instructions == identity["allowed_instructions"] and states.shape == (3, 1, 7)
    assert states[0, 0, 6] != .02 / 8


def test_old_single_task_contract_cannot_load_as_multitask(tmp_path):
    run = fixture(tmp_path, "near_multitask_joint6_3000")
    contract = json.loads((run / "training_contract.json").read_text())
    contract["contract"].update(source_episode_count=1000, split_counts=dict(train=900, val=100))
    (run / "training_contract.json").write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="3000-episode"):
        load_metadata(run)
