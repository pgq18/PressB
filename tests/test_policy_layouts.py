"""Fixed layout coverage, seed stability and physical reset interface without Isaac."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pressb.policy_layouts import (LAYOUT_FIELDS, evaluation_panel_layouts, evaluation_schedule,
                                   reposition_evaluation_panels, evaluation_layout_evidence)


ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/dataset_panel_stratified_1200.json").read_text())


def test_center_corners_uses_recorded_rectangle_without_mutating_config():
    before = deepcopy(CFG)
    layouts = evaluation_panel_layouts(CFG, "center_corners")
    assert CFG == before
    assert [row["panel_layout_name"] for row in layouts] == ["center", "xmin_ymin", "xmin_ymax", "xmax_ymin", "xmax_ymax"]
    assert [(row["panel_offset_x_m"], row["panel_offset_y_m"]) for row in layouts] == [
        (0., 0.), (-.01, -.025), (-.01, .025), (.01, -.025), (.01, .025)]
    assert all(set(row) == set(LAYOUT_FIELDS) for row in layouts)


def test_asymmetric_rectangle_uses_actual_midpoint():
    cfg = deepcopy(CFG)
    cfg["panel_randomization"].update(min_offset_x_m=-.006, max_offset_x_m=.01,
                                      min_offset_y_m=-.012, max_offset_y_m=.022)
    center = evaluation_panel_layouts(cfg, "center_corners")[0]
    assert center["panel_offset_x_m"] == pytest.approx(.002)
    assert center["panel_offset_y_m"] == pytest.approx(.005)


def test_sixty_conditions_cover_each_floor_and_layout_once():
    layouts = evaluation_panel_layouts(CFG, "center_corners")
    rows = evaluation_schedule(range(24, 36), 1, layouts)
    assert len(rows) == 60
    assert [row["episode_id"] for row in rows] == list(range(60))
    assert [row["seed_episode_index"] for row in rows] == list(range(60))
    for floor in range(24, 36):
        selected = [row for row in rows if row["floor"] == floor]
        assert [row["panel_layout_index"] for row in selected] == list(range(5))
        assert {row["panel_layout_name"] for row in selected} == {p["panel_layout_name"] for p in layouts}


def test_four_workers_and_floor_order_preserve_conditions_and_seed_streams():
    layouts = evaluation_panel_layouts(CFG, "center_corners")
    full = evaluation_schedule(range(24, 36), 2, layouts)
    by_condition = {(row["repeat"], row["panel_layout_index"], row["floor"]): row for row in full}
    partitioned = []
    for first in range(24, 36, 3):
        partitioned.extend(evaluation_schedule(range(first, first + 3), 2, layouts))
    assert len(partitioned) == len(full) == 120
    assert len({r["seed_episode_index"] for r in partitioned}) == 120
    for row in partitioned + evaluation_schedule([35, 24, 28], 2, layouts):
        original = by_condition[row["repeat"], row["panel_layout_index"], row["floor"]]
        assert {k: v for k, v in row.items() if k != "episode_id"} == {
            k: v for k, v in original.items() if k != "episode_id"}
    # The final partial batch must keep its requested layout/floor pairing.
    assert [(r["panel_layout_name"], r["floor"]) for r in partitioned[-2:]] == [("xmax_ymax", 34), ("xmax_ymax", 35)]


def test_fixed_mode_preserves_legacy_order_and_seed_for_subsets():
    layouts = evaluation_panel_layouts({})
    assert layouts == [dict(panel_layout_index=0, panel_layout_name="fixed", panel_offset_x_m=0., panel_offset_y_m=0.)]
    rows = evaluation_schedule([35, 24], 3, layouts)
    assert [(r["floor"], r["repeat"]) for r in rows] == [(f, repeat) for repeat in range(3) for f in [35, 24]]
    assert [r["seed_episode_index"] for r in rows] == [11, 0, 23, 12, 35, 24]
    shifted = evaluation_panel_layouts(dict(panel_offset_x_m=.002, panel_offset_y_m=-.001))
    assert shifted[0]["panel_offset_x_m"] == .002 and shifted[0]["panel_offset_y_m"] == -.001


@pytest.mark.parametrize("change", ["disabled", "missing", "nonfinite", "boolean", "collapsed", "reversed"])
def test_unsafe_or_undefined_rectangle_is_rejected_before_simulation(change):
    cfg = deepcopy(CFG)
    settings = cfg["panel_randomization"]
    if change == "disabled":
        settings["enabled"] = False
    elif change == "missing":
        del settings["max_offset_x_m"]
    elif change == "nonfinite":
        settings["max_offset_y_m"] = float("nan")
    elif change == "boolean":
        settings["min_offset_y_m"] = False
    elif change == "collapsed":
        settings["max_offset_x_m"] = settings["min_offset_x_m"]
    else:
        settings["max_offset_x_m"] = settings["min_offset_x_m"] - .01
    with pytest.raises(ValueError):
        evaluation_panel_layouts(cfg, "center_corners")


@pytest.mark.parametrize("floors,repeats", [([], 1), ([24, 24], 1), ([23], 1), ([36], 1),
                                         ([True], 1), ([24.0], 1), ([24], 0), ([24], True)])
def test_invalid_schedule_rejected(floors, repeats):
    with pytest.raises(ValueError):
        evaluation_schedule(floors, repeats, evaluation_panel_layouts({}))


def test_panel_reset_uses_absolute_xy_and_never_steps_or_touches_inactive_env(monkeypatch):
    from pressb import dataset_scene

    calls = []
    world = object()  # No step or render API exists: this helper must only edit.
    envs = [SimpleNamespace(env_id=i) for i in range(3)]
    monkeypatch.setattr(dataset_scene, "set_panel_offset",
                        lambda w, env, **kw: calls.append((w, env.env_id, kw)))
    rows = evaluation_schedule([24], 1, evaluation_panel_layouts(CFG, "center_corners"))
    reposition_evaluation_panels(world, envs, rows[1:3])
    reposition_evaluation_panels(world, envs, rows[3:4])
    assert calls == [(world, 0, {"offset_y_m": -.025, "offset_x_m": -.01}),
                     (world, 1, {"offset_y_m": .025, "offset_x_m": -.01}),
                     (world, 0, {"offset_y_m": -.025, "offset_x_m": .01})]
    assert all(eid != 2 for _, eid, _ in calls)
    with pytest.raises(ValueError, match="More evaluation layouts"):
        reposition_evaluation_panels(world, envs, rows)


@pytest.mark.parametrize("bad_axis", [None, "x", "y"])
def test_live_layout_proof_rejects_stale_offsets_before_capture(monkeypatch, bad_axis):
    from pressb import dataset_scene

    episode = evaluation_panel_layouts(CFG, "center_corners")[4]
    proof = dict(panel_offset_x_m=.01, panel_offset_y_m=.025, fixed_rig_unchanged=True,
                 button_rest_positions_world_m={"24": [.473, 6.07, .98]})
    if bad_axis:
        proof[f"panel_offset_{bad_axis}_m"] = 0.
    monkeypatch.setattr(dataset_scene, "validate_panel_layout", lambda w, env: proof)
    if bad_axis:
        with pytest.raises(RuntimeError, match="wrong .* offset"):
            evaluation_layout_evidence(object(), object(), episode)
    else:
        assert evaluation_layout_evidence(object(), object(), episode) is proof
