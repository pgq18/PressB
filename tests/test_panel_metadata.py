"""Randomized panel context cannot be lost or silently interpreted at zero."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

from pressb.panel_metadata import episode_panel_context, validate_episode_panel_metadata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_lerobot import audit_episode_collection
from export_lerobot import EPISODE_CONTEXT, entry_provenance, feature_spec
from audit_raw_dataset import sampling_parameters


def panel_fixture():
    x, y = .005, -.012
    cfg = dict(button_face_x=.46, button_column_y=.045, button_bottom_z=.98,
               button_pitch_z=.035, physics_dt=1 / 120, panel_offset_x_m=0., panel_offset_y_m=0.,
               panel_randomization=dict(enabled=True, min_offset_x_m=-.01, max_offset_x_m=.01,
                                        min_offset_y_m=-.03, max_offset_y_m=.03, camera_margin_px=5., max_attempts=64))
    collection = dict(raw_schema_version=11, fps=30, physics_hz=120, capture_stride=4,
                      action_horizon_s=1 / 30, collection_fingerprint="fixture", config=cfg)
    layout = dict(panel_offset_x_m=x, panel_offset_y_m=y,
                  button_rest_positions_world_m={str(floor): [.463 + x, 6 + y + (.045 if floor < 30 else -.045),
                                                              .98 + ((floor - 24) % 6) * .035] for floor in range(24, 36)},
                  max_button_rest_error_m=0., max_spring_anchor_error_m=0., fixed_rig_unchanged=True,
                  panel_world_bounds_m=[[.455 + x, 6 + y - .1, .91], [.509 + x, 6 + y + .1, 1.225]],
                  global_camera=dict(path="/env/global", resolution=[640, 480], margin_px=5., all_inside=True,
                                     corners_px=[[200., 120.], [200., 300.], [350., 120.], [350., 300.]] * 2,
                                     corner_depths_m=[.7] * 8, coverage_kind="frustum_only"))
    metadata = {key: value for key, value in collection.items() if key != "config"}
    metadata.update(panel_offset_x_m=x, panel_offset_y_m=y, panel_layout=layout,
                    env_offset_m=[0., 6., 0.], robot_base_world_m=[-.24, 6., .76],
                    variation=dict(panel_offset_x_m=x, panel_offset_y_m=y,
                                   panel_randomization={**cfg["panel_randomization"], "panel_offset_x_m": x, "panel_offset_y_m": y}))
    return collection, metadata


@pytest.mark.parametrize("schema", [7, 9, 10])
def test_legacy_context_remains_absent_and_means_zero(schema):
    collection = {"raw_schema_version": schema, "config": {}}
    metadata = {"raw_schema_version": schema}
    assert validate_episode_panel_metadata(collection, metadata) == (0., 0.)
    assert episode_panel_context(metadata) == {}
    assert EPISODE_CONTEXT == ("raw_schema_version", "collection_fingerprint", "env_offset_m", "robot_base_world_m")


def test_schema11_retains_both_offsets_and_measured_layout(tmp_path):
    collection, metadata = panel_fixture()
    assert validate_episode_panel_metadata(collection, metadata) == (.005, -.012)
    assert sampling_parameters(collection, metadata)["capture_stride"] == 4
    episode = dict(directory=tmp_path, episode_id=0, floor=24, seed=2026, metadata=metadata)
    for name in ("metadata.json", "frames.npz", "physics.npz", "wrist.mp4", "global.mp4"):
        (tmp_path / name).write_bytes(name.encode())
    entry = entry_provenance(episode, 9)
    audit_episode_collection(entry, metadata, collection, tmp_path)
    assert episode_panel_context(entry) == episode_panel_context(metadata)
    assert feature_spec()["observation.state"]["shape"] == (8,)
    assert feature_spec()["action"]["shape"] == (8,)
    assert "panel_offset_x_m" not in feature_spec()
    for key in ("panel_offset_x_m", "panel_offset_y_m", "panel_layout"):
        changed = deepcopy(entry)
        changed.pop(key)
        with pytest.raises(ValueError, match="differs from the raw metadata"):
            audit_episode_collection(changed, metadata, collection, tmp_path)


@pytest.mark.parametrize("change", ["missing_x", "missing_y", "outside_x", "outside_y", "nonfinite", "variation", "layout",
                                    "nominal_rest_x", "nominal_rest_y", "wrong_env", "camera_margin", "camera_depth", "moved_rig"])
def test_schema11_rejects_incomplete_or_contradictory_layout(change):
    collection, metadata = panel_fixture()
    if change.startswith("missing_"):
        metadata.pop(f"panel_offset_{change[-1]}_m")
    elif change.startswith("outside_"):
        metadata[f"panel_offset_{change[-1]}_m"] = .3
    elif change == "nonfinite":
        metadata["panel_offset_y_m"] = float("nan")
    elif change == "variation":
        metadata["variation"]["panel_offset_y_m"] = 0.
    elif change == "layout":
        metadata.pop("panel_layout")
    elif change.startswith("nominal_rest_"):
        axis = 0 if change[-1] == "x" else 1
        metadata["panel_layout"]["button_rest_positions_world_m"]["24"][axis] -= metadata[f"panel_offset_{change[-1]}_m"]
    elif change == "wrong_env":
        metadata["env_offset_m"] = [0., 0., 0.]
    elif change == "camera_margin":
        metadata["panel_layout"]["global_camera"]["corners_px"][0][0] = 2.
    elif change == "camera_depth":
        metadata["panel_layout"]["global_camera"]["corner_depths_m"][0] = -1.
    elif change == "moved_rig":
        metadata["panel_layout"]["fixed_rig_unchanged"] = False
    with pytest.raises(ValueError):
        validate_episode_panel_metadata(collection, metadata)


def test_camera_bezel_geometry_uses_both_episode_axes():
    pytest.importorskip("cv2")
    pytest.importorskip("av")
    from audit_camera_sync import bezel_points
    collection, _ = panel_fixture()
    nominal = bezel_points(collection["config"])
    shifted = bezel_points(collection["config"], panel_offset_x_m=.005, panel_offset_y_m=-.012)
    np.testing.assert_allclose(shifted - nominal, np.broadcast_to([.005, -.012, 0.], nominal.shape), atol=1e-15)
