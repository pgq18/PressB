"""One layout per stratum, corner coverage and immutable retry assignments."""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path

import numpy as np
import pytest

from pressb.dataset_planning import make_episode_plan
from pressb.kinematics import PiperKinematics
from pressb.panel_metadata import validate_episode_panel_metadata
from pressb.panel_randomization import panel_randomization_settings, sample_panel_layout


ROOT = Path(__file__).resolve().parents[1]
BASE = json.loads((ROOT / "configs/dataset_panel_randomized.json").read_text())
SEED = 20260930


def config(**overrides):
    return dict(BASE, panel_randomization={**BASE["panel_randomization"],
        "sampling_mode": "stratified_grid", "grid_shape": [10, 10], "grid_seed": SEED,
        "boundary_mode": "corners", **overrides})


def safe_layout(*args, **kwargs):
    return {"all_buttons_reachable": True, "reachable_floors": list(range(24, 36))}


def sample(cfg, floor, ordinal):
    return sample_panel_layout(None, cfg, floor, SEED + ordinal * 12 + floor - 24, episode_index=ordinal)


def test_every_floor_covers_all_100_cells_and_four_exact_corners(monkeypatch):
    # Stub expensive safety checks here: this test measures the layout design,
    # while separate trajectory tests and runtime validation exercise physics.
    calls = []
    monkeypatch.setattr("pressb.panel_randomization.validate_panel_layout",
                        lambda *a, **kw: calls.append(1) or safe_layout())
    cfg = config()
    bounds = cfg["panel_randomization"]
    edges = {axis: np.linspace(bounds[f"min_offset_{axis}_m"], bounds[f"max_offset_{axis}_m"], 11) for axis in "xy"}
    permutations = []
    for floor in range(24, 36):
        records = [sample(cfg, floor, index) for index in range(100)]
        cells, corners = [], []
        for index, (layout, proof) in enumerate(records):
            xy = [layout[f"panel_offset_{axis}_m"] for axis in "xy"]
            actual_cell = tuple(min(9, int(np.searchsorted(edges[axis], value, side="right") - 1))
                                for axis, value in zip("xy", xy))
            assert actual_cell == tuple(proof["grid_cell_index_xy"])
            assert proof["episode_index"] == index
            cells.append(actual_cell)
            if proof["boundary_corner"]:
                corners.append(tuple(xy))
            else:
                for axis, value in zip("xy", xy):
                    low, high = proof["grid_cell_bounds_m"][axis]
                    assert low <= value < high
        assert set(cells) == set(product(range(10), repeat=2))
        assert len(set(cells)) == 100
        assert set(corners) == set(product([-.01, .01], [-.025, .025]))
        assert len(corners) == 4
        permutations.append(tuple(cells))
        for index in (99, 0, 37):
            assert sample(cfg, floor, index) == records[index]
    assert len(set(permutations)) == 12
    assert len(calls) == 12 * 103


def test_safety_retries_never_leave_the_assigned_cell_and_exhaustion_fails(monkeypatch):
    monkeypatch.setattr("pressb.panel_randomization.validate_panel_layout", safe_layout)
    cfg = config(max_attempts=3)
    index = next(i for i in range(100) if not sample(cfg, 24, i)[1]["boundary_corner"])
    _, reference = sample(cfg, 24, index)
    calls = []

    def third_pass(kin, episode_cfg, margin):
        calls.append([episode_cfg[f"panel_offset_{a}_m"] for a in "xy"])
        if len(calls) % 3:
            raise ValueError("deliberate rejected trajectory")
        return safe_layout()

    monkeypatch.setattr("pressb.panel_randomization.validate_panel_layout", third_pass)
    result, proof = sample(cfg, 24, index)
    assert proof["grid_cell_index_xy"] == reference["grid_cell_index_xy"]
    assert proof["attempts"] == 3 and len(proof["rejected_candidates"]) == 2
    for xy in calls:
        for axis, value in zip("xy", xy):
            low, high = proof["grid_cell_bounds_m"][axis]
            assert low <= value < high
    assert len(set(map(tuple, calls))) == 3
    with pytest.raises(ValueError, match="No safe panel layout after 2 attempts"):
        sample(config(max_attempts=2), 24, index)


def test_unsafe_required_corner_is_not_replaced_by_an_interior_sample(monkeypatch):
    monkeypatch.setattr("pressb.panel_randomization.validate_panel_layout", safe_layout)
    cfg = config()
    index = next(i for i in range(100) if sample(cfg, 24, i)[1]["boundary_corner"])
    calls = []

    def unsafe(*args):
        calls.append(1)
        raise ValueError("corner unreachable")

    monkeypatch.setattr("pressb.panel_randomization.validate_panel_layout", unsafe)
    with pytest.raises(ValueError, match="Required panel corner"):
        sample(cfg, 24, index)
    assert len(calls) == 1


def test_legacy_uniform_draws_and_metadata_remain_identical(monkeypatch):
    monkeypatch.setattr("pressb.panel_randomization.validate_panel_layout", safe_layout)
    for floor, seed in [(24, 0), (29, 134), (35, 20262129)]:
        layout, proof = sample_panel_layout(None, BASE, floor, seed)
        explicit = dict(BASE, panel_randomization={**BASE["panel_randomization"], "sampling_mode": "uniform"})
        _, repeated = sample_panel_layout(None, explicit, floor, seed, episode_index=99)
        assert proof == repeated
        rng = np.random.default_rng(np.random.SeedSequence([seed, floor, 0x50414E45]))
        assert layout["panel_offset_x_m"] == rng.uniform(-.01, .01)
        assert layout["panel_offset_y_m"] == rng.uniform(-.025, .025)
        assert "sampling_mode" not in proof and "grid_cell_index_xy" not in proof


@pytest.mark.parametrize("index", [None, -1, 100, 1., True])
def test_missing_or_invalid_floor_episode_index_rejected(index):
    with pytest.raises(ValueError, match="episode_index"):
        sample_panel_layout(None, config(), 24, SEED, episode_index=index)


@pytest.mark.parametrize("changes", [
    {"sampling_mode": "unknown"}, {"grid_shape": [10]}, {"grid_shape": [0, 10]},
    {"grid_shape": [10., 10]}, {"grid_shape": [True, 10]}, {"grid_shape": [101, 100]},
    {"grid_seed": -1}, {"grid_seed": True}, {"boundary_mode": "ignored"},
    {"grid_shape": [1, 10]}, {"min_offset_x_m": .01, "max_offset_x_m": .01},
])
def test_invalid_grid_settings_rejected(changes):
    with pytest.raises(ValueError):
        panel_randomization_settings(config(**changes))


def metadata_fixture(monkeypatch, corner=False):
    monkeypatch.setattr("pressb.panel_randomization.validate_panel_layout", safe_layout)
    cfg = config()
    ordinal = next(i for i in range(100) if sample(cfg, 24, i)[1]["boundary_corner"] is corner)
    layout_cfg, proof = sample(cfg, 24, ordinal)
    x, y = layout_cfg["panel_offset_x_m"], layout_cfg["panel_offset_y_m"]
    layout = dict(panel_offset_x_m=x, panel_offset_y_m=y, fixed_rig_unchanged=True,
        max_button_rest_error_m=0., max_spring_anchor_error_m=0.,
        button_rest_positions_world_m={str(f): [cfg["button_face_x"] + .003 + x,
            cfg["button_column_y"] * (1 if f < 30 else -1) + y,
            cfg["button_bottom_z"] + ((f - 24) % 6) * cfg["button_pitch_z"]] for f in range(24, 36)},
        panel_world_bounds_m=[[.4597 + x, y - .1, .91], [.509 + x, y + .1, 1.225]],
        global_camera=dict(resolution=[640, 480], margin_px=24., all_inside=True,
            corners_px=[[200., 120.], [200., 300.], [350., 120.], [350., 300.]] * 2,
            corner_depths_m=[.7] * 8))
    metadata = dict(episode_id=ordinal * 12, floor=24, seed=SEED + ordinal * 12,
        panel_offset_x_m=x, panel_offset_y_m=y, env_offset_m=[0., 0., 0.], panel_layout=layout,
        variation=dict(panel_offset_x_m=x, panel_offset_y_m=y, panel_randomization=proof))
    return dict(raw_schema_version=11, config=cfg), metadata


@pytest.mark.parametrize("corner", [False, True])
def test_stratified_metadata_geometry_and_identity_validated(monkeypatch, corner):
    collection, metadata = metadata_fixture(monkeypatch, corner)
    assert validate_episode_panel_metadata(collection, metadata) == (metadata["panel_offset_x_m"], metadata["panel_offset_y_m"])


@pytest.mark.parametrize("change", ["mode", "shape", "seed", "index", "flat", "bounds", "cell", "corner", "retry"])
def test_contradictory_stratum_metadata_rejected(monkeypatch, change):
    collection, metadata = metadata_fixture(monkeypatch)
    proof = metadata["variation"]["panel_randomization"]
    if change == "mode": proof["sampling_mode"] = "uniform"
    elif change == "shape": proof["grid_shape"] = [5, 20]
    elif change == "seed": proof["grid_seed"] += 1
    elif change == "index": proof["episode_index"] += 1
    elif change == "flat": proof["grid_flat_index"] = (proof["grid_flat_index"] + 1) % 100
    elif change == "bounds": proof["grid_cell_bounds_m"]["x"][0] -= .001
    elif change == "cell": proof["grid_cell_index_xy"][0] = (proof["grid_cell_index_xy"][0] + 1) % 10
    elif change == "corner": proof["boundary_corner"] = True
    elif change == "retry":
        proof["attempts"] = 2
        proof["rejected_candidates"] = [dict(panel_offset_x_m=.03, panel_offset_y_m=0., reason="outside cell")]
    with pytest.raises(ValueError):
        validate_episode_panel_metadata(collection, metadata)


def test_dataset_planner_uses_stratified_layout_and_records_floor_ordinal():
    cfg = config()
    kin = PiperKinematics(ROOT / cfg["robot_urdf"], tip_offset=cfg["tip_offset"])
    plan = make_episode_plan(kin, cfg, 24, SEED, episode_index=0)
    proof = plan.metadata["panel_randomization"]
    assert proof["episode_index"] == 0 and proof["sampling_mode"] == "stratified_grid"
    assert proof["validation"]["reachable_floors"] == list(range(24, 36))
    hold = plan.phase == "hold"
    np.testing.assert_allclose(plan.target_tip[hold, 0], cfg["button_face_x"] + plan.metadata["panel_offset_x_m"] + cfg["press_depth"])
    np.testing.assert_allclose(plan.target_tip[hold, 1], cfg["button_column_y"] + plan.metadata["panel_offset_y_m"])
