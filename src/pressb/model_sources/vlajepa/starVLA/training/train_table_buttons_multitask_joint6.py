"""Train one 3,000-episode multi-task model with joint7 state/joint6 actions.

This entrypoint shares the tested optimizer, initialization and checkpoint loop
with train_table_buttons. Only data splitting and balanced validation differ.
"""
from __future__ import annotations

import hashlib
import json
import time
import numpy as np
import torch
import torch.distributed as dist

from starVLA.training import train_piper as shared
from starVLA.training import train_table_buttons as single

COLORS = ("red", "yellow", "blue")


def validate_anchor_balance(dataset, anchors):
    expected = {color: 500 for color in COLORS}
    actual = {color: 0 for color in COLORS}
    for episode, frame in anchors:
        actual[dataset._episodes[episode]["target_color"]] += 1
    if len(anchors) != 1500 or len(set(anchors)) != 1500 or actual != expected:
        raise ValueError("Expected 1,500 distinct validation anchors, 100 episodes x 5 per color")


def summarize_multitask_metrics(totals):
    totals = np.asarray(totals, dtype=np.float64)
    if totals.shape != (3, 12) or not np.isfinite(totals).all() or np.any(totals[:, 0] <= 0):
        raise ValueError("Expected finite nonempty validation totals for all three colors")
    def result(row):
        metrics = single.summarize_joint_metrics(row, action_dim=6)
        metrics.update(flow_loss=float(row[8] / max(row[9], 1)),
                       wm_loss_weighted_0_1=float(row[10] / max(row[11], 1)))
        return metrics
    per_color = {color: result(totals[index]) for index, color in enumerate(COLORS)}
    overall = result(totals.sum(axis=0))
    overall["selection_score"] = float(np.mean([value["first_joint_mae_deg"] for value in per_color.values()]))
    overall["selection_aggregation"] = "equal-weight mean of three per-color first-action six-joint MAEs"
    return overall, per_color


@torch.no_grad()
def evaluate(model, dataset, anchors, rank, world, device, seed, step, output):
    model.eval()
    totals = np.zeros((3, 12), dtype=np.float64)
    started = time.monotonic()
    if rank == 0:
        print(json.dumps(dict(event="validation_started", step=step, anchors=len(anchors))), flush=True)
    for ordinal in range(rank, len(anchors), world):
        episode, frame = anchors[ordinal]
        example = dataset.get_frame(episode, frame)
        color = COLORS.index(example["target_color"])
        layout = dataset._episodes[episode]["entry"]["layout_group_id"]
        # Stable shared noise for identical layouts, regardless of retry/source IDs.
        example_seed = (seed + int(layout[:12], 16) * 100003 + frame) % (2**63 - 1)
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(example_seed)
            torch.cuda.manual_seed(example_seed)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predicted = model.predict_action([example["image"]], [example["lang"]],
                    state=np.stack([example["state"]]))["normalized_actions"][0]
                losses = model([example])
        values = single.joint_metric_sums(predicted, example["action"], example["state"],
                                         example["action_is_pad"], action_dim=6)
        future = (~example["video_is_pad"].reshape(-1, model.vj_encoder.config.tubelet_size).any(-1))[1:].sum()
        totals[color, :8] += values
        totals[color, 8:] += [float(losses["action_loss"]) * values[1], values[1],
                              float(losses["wm_loss"]) * future, future]
    reduced = torch.as_tensor(totals, device=device)
    dist.all_reduce(reduced)
    overall, per_color = summarize_multitask_metrics(reduced.cpu().numpy())
    if overall["anchors"] != len(anchors) or any(m["anchors"] != 500 for m in per_color.values()):
        raise RuntimeError("Validation skipped or repeated per-color anchors")
    result = dict(step=step, evaluation_seed=seed, seconds=time.monotonic() - started,
                  overall=overall, per_color=per_color, group=dataset.group,
                  selection="equal-color mean of first_joint_mae_deg (six arm joints; gripper excluded)",
                  interpretation="offline held-out-layout action prediction; not simulator task success rate")
    if rank == 0:
        shared.write_json(output / f"validation_{step:06d}.json", result)
        print(json.dumps(dict(event="validation", step=step, **overall, per_color=per_color)), flush=True)
    return result


def build_contract(cfg, normalization, split_manifest):
    if cfg["framework"]["action_model"]["action_dim"] != 6 or cfg["framework"]["action_model"]["state_dim"] != 7:
        raise ValueError("Multi-task joint6 training requires state7/action6")
    if split_manifest["counts"] != {"train": 2700, "val": 300}:
        raise ValueError("Expected 2,700 train and 300 held-out episodes")
    contract, _ = shared.training_contract(cfg)
    contract.update(schema=single.JOINT6_SCHEMA, multitask=True, normalization=normalization,
                    target_colors=list(COLORS), checkpoint_policy=single.configured_checkpoint_policy(cfg),
                    validation_selection_weight=0., validation_selection_aggregation="equal weight per color",
                    validation_anchors_per_color=500, validation_anchor_count=1500,
                    source_episode_count=3000, split_counts=split_manifest["counts"],
                    split_rule=split_manifest["selection"], layout_group_key="layout_group_id",
                    split_layout_counts=split_manifest["layout_group_counts"],
                    excluded_evaluation_seed_range=[10000, 10099],
                    language="per-episode Press the <color> button., identical across distance profiles",
                    initialization="official pretrained, four joint6 I/O tensors reset")
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return contract, fingerprint


def main():
    from starVLA.dataloader.table_buttons_multitask_joint6_lerobot import TableButtonsMultitaskJoint6LeRobotDataset
    single.main(dataset_class=TableButtonsMultitaskJoint6LeRobotDataset, evaluator=evaluate,
                contract_builder=build_contract, anchor_validator=validate_anchor_balance)


if __name__ == "__main__":
    main()
