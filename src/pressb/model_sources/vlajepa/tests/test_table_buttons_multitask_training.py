"""CPU tests for balanced multi-task validation and independent training contracts."""
from types import SimpleNamespace
import numpy as np
import pytest

from starVLA.training.train_table_buttons_multitask import (
    COLORS, summarize_multitask_metrics, validate_anchor_balance, build_contract,
)


def test_multitask_aggregate_and_task_balanced_best_selection():
    totals = np.zeros((3, 12), dtype=float)
    for i, n in enumerate((1, 2, 4)):
        totals[i, 0] = n
        totals[i, 1] = n * 3
        totals[i, 2] = 6 * n * 3 * (i + 1)
        totals[i, 3] = n * 3 * .2
        totals[i, 4] = 6 * n * (i + 1)
        totals[i, 5] = n * .4
        totals[i, 6] = 6 * n * 10
        totals[i, 7] = n * .1
        totals[i, 8:12] = [n * .5, n, n * .6, n]
    overall, per_color = summarize_multitask_metrics(totals)
    assert overall["anchors"] == 7 and overall["valid_actions"] == 21
    assert set(per_color) == set(COLORS)
    assert overall["selection_score"] == pytest.approx(2 + 22.5 * .4)
    # Aggregate MAE remains sample weighted; best-model selection is color balanced.
    assert overall["first_joint_mae_deg"] == pytest.approx(17 / 7)
    assert per_color["red"]["first_joint_mae_deg"] == 1
    assert per_color["yellow"]["first_joint_mae_deg"] == 2
    assert per_color["blue"]["first_joint_mae_deg"] == 3
    assert overall["state_copy_baseline_first_joint_mae_deg"] == 10
    assert overall["flow_loss"] == pytest.approx(.5)


@pytest.mark.parametrize("bad", [np.zeros((3, 12)), np.ones((2, 12)), np.full((3, 12), np.nan)])
def test_missing_color_or_nonfinite_metrics_rejected(bad):
    with pytest.raises(ValueError):
        summarize_multitask_metrics(bad)


def test_anchor_balance_requires_all_three_tasks_and_no_duplicates():
    episodes = {i: {"target_color": COLORS[i // 10]} for i in range(30)}
    dataset = SimpleNamespace(_episodes=episodes)
    anchors = [(i, frame) for i in range(30) for frame in range(5)]
    validate_anchor_balance(dataset, anchors)
    for bad in (anchors[:-1], anchors[:-1] + [anchors[0]], anchors[:100] + [(0, i) for i in range(50, 100)]):
        with pytest.raises(ValueError):
            validate_anchor_balance(dataset, bad)


def test_contract_records_layout_split_language_and_balanced_selection():
    # training_contract includes architecture, optimizer and cadence keys supplied by the actual config.
    import yaml
    from pathlib import Path
    config_path = Path(__file__).resolve().parents[1] / "scripts/configs/vlajepa_piper_ft.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    cfg["framework"]["action_model"].update(state_dim=7, action_dim=7)
    cfg["trainer"]["target_epochs"] = 3
    contract, fingerprint = build_contract(cfg, {"kind": "fixed_joint7_scale", "scale": [180.] * 6 + [8.]})
    assert contract["multitask"] is True
    assert contract["split_counts"] == {"train": 270, "val": 30}
    assert contract["validation_anchors_per_color"] == 50
    assert contract["validation_selection_aggregation"] == "equal weight per color"
    assert "all colors" in contract["split_rule"]
    assert len(fingerprint) == 64
