"""Joint6 multi-task metadata with the existing six-action inference implementation."""
from pathlib import Path

from starVLA.inference.table_buttons_joint6_policy import (
    TableButtonsJoint6Policy, NATIVE_SCHEMA, CAMERAS, WIRE_SCALE, WIRE_SCHEMA,
    ACTION_SCALE, validate_normalization, read_json,
)

COLORS = ("red", "yellow", "blue")
MODEL_GROUPS = {f"{spacing}_multitask_joint6_3000": f"{spacing}_multitask"
                for spacing in ("near", "medium", "far")}


def load_metadata(run, model_id=None, checkpoint=None):
    """Read native metadata; archived or in-run checkpoints are both supported.

    This does not read the full model. A missing published digest is computed
    once during actual model loading and returned to the client before use.
    """
    from omegaconf import OmegaConf
    run = Path(run).resolve(strict=True)
    config = OmegaConf.load(run / "config.yaml")
    schema, normalization = read_json(run / "action_schema.json"), read_json(run / "normalization.json")
    checkpoint = Path(checkpoint).resolve(strict=True) if checkpoint else (run / "checkpoints/best").resolve(strict=True)
    model_path = checkpoint / "model.pt"
    manifest = read_json(checkpoint / "manifest.json")
    model_id = str(config.run_id) if model_id is None else model_id
    if model_id not in MODEL_GROUPS or config.run_id != model_id:
        raise ValueError("Unknown or mismatched multi-task joint6 run")
    group = MODEL_GROUPS[model_id]
    if schema.get("schema") != NATIVE_SCHEMA or manifest.get("schema") != NATIVE_SCHEMA:
        raise ValueError("Expected native joint6 schema in action and checkpoint manifests")
    for key, value in (("state_dim", 7), ("action_dim", 6), ("action_horizon", 7)):
        if schema.get(key) != value or int(getattr(config.framework.action_model, key)) != value:
            raise ValueError(f"Native model architecture mismatch: {key}")
    if schema.get("cameras") != list(CAMERAS) or schema.get("fps") != 30:
        raise ValueError("Expected up/wrist cameras at 30 Hz")
    width = validate_normalization(normalization)
    if schema.get("normalization") != normalization or schema.get("fixed_gripper_full_width_m") != width:
        raise ValueError("Action schema and fixed normalization differ")
    if not manifest.get("complete") or model_path.stat().st_size != manifest["model_bytes"]:
        raise ValueError("Checkpoint is incomplete or has an unexpected size")
    for key in ("dataset_sha256", "split_sha256", "training_fingerprint"):
        if not schema.get(key) or schema[key] != manifest.get(key):
            raise ValueError(f"Checkpoint provenance mismatch: {key}")
    if schema.get("dataset_group") != group:
        raise ValueError("Joint6 model dataset group differs from its run ID")
    data_root = Path(config.datasets.vla_data.data_root_dir)
    if data_root.name != group:
        raise ValueError("Configuration dataset group differs from the joint6 model")
    # Count provenance remains in the saved training contract; inference
    # does not require the original training dataset to remain mounted.
    contract = read_json(run / "training_contract.json")["contract"]
    if contract.get("source_episode_count") != 3000 or contract.get("split_counts") != {"train": 2700, "val": 300} or contract.get("layout_group_key") != "layout_group_id" or contract.get("multitask") is not True:
        raise ValueError("Multi-task joint6 model must retain the 3000-episode 2700/300 layout-grouped training contract")
    digest = manifest.get("model_sha256")
    archive_path = run / "archive.json"
    if archive_path.is_file():
        archive = read_json(archive_path)
        # An explicit last/probe checkpoint need not be the archived best model.
        if Path(archive["model_path"]).resolve() == model_path.resolve():
            if archive["model_bytes"] != model_path.stat().st_size or archive["best_step"] != manifest["step"]:
                raise ValueError("Archived model metadata differs from selected checkpoint")
            if digest and digest != archive["model_sha256"]:
                raise ValueError("Archive and checkpoint digests differ")
            digest = archive["model_sha256"]
    identity = dict(model_id=model_id, spacing=group.split("_")[0], task_type="multitask", source_dataset_group=group,
        source_episode_count=3000,
        allowed_instructions=[f"Press the {color} button." for color in COLORS], model_sha256=digest,
        model_bytes=model_path.stat().st_size, best_step=manifest["step"], checkpoint_step=manifest["step"],
        checkpoint_path=str(model_path), run=str(run), camera_order=list(CAMERAS), image_input_size=[640, 480],
        image_resize="PIL RGB BILINEAR square; identical to training adapter",
        image_model_size=[int(config.datasets.vla_data.resolution_size)] * 2,
        state_units=["degree"] * 6 + ["full_width_cm"], action_units=["degree"] * 6 + ["full_width_cm"],
        normalization_scale=WIRE_SCALE.tolist(), state_dim=7, action_dim=7, action_horizon=7,
        wire_schema=WIRE_SCHEMA, wire_action_dim=7, native_schema=NATIVE_SCHEMA, native_action_dim=6,
        noise_shape_per_sample=[1, 7, 6], noise_hash_array_shape=[7, 6], noise_hash_dtype="float32",
        noise_response_field="initial_noise", noise_hash_memory_order="C",
        native_action_units=["degree"] * 6, native_action_normalization_scale=ACTION_SCALE.tolist(),
        action_dim_semantics="action_dim is the transport width; the network has native_action_dim=6",
        fixed_gripper_full_width_m=width, fixed_gripper_full_width_cm=width * 100., gripper_learned=False,
        gripper_rule="append the fixed dataset full-width command after native six-arm-action prediction",
        fps=30, smoothing="none", action_clipping="none", recurrent_state="none",
        num_inference_timesteps=int(config.framework.action_model.num_inference_timesteps))
    return config, identity


class TableButtonsMultitaskJoint6Policy(TableButtonsJoint6Policy):
    metadata_loader = staticmethod(load_metadata)
