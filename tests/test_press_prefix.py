"""Causal terminal labels and negative physical/capture evidence tests."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pressb.kinematics import PiperKinematics
from pressb.press_prefix import crop_press_prefix_arrays, inspect_press_prefix_arrays, tcp_transforms


ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/scene.json").read_text())
URDF = ROOT / CFG["robot_urdf"]


def pose(kin, q):
    result = kin.fk(q)
    return np.r_[result[:3, 3], Rotation.from_matrix(result[:3, :3]).as_quat()[[3, 0, 1, 2]], .008]


@pytest.fixture
def source():
    kin = PiperKinematics(URDF, tip_offset=.1358)
    q = np.array([.2, .7, -1., .25, .3, -.2]) + np.arange(65)[:, None] * np.array([.001, -.001, .001, .002, -.001, .001])
    actual = q + [.0001, -.0002, .0001, 0., .0001, 0.]
    lamps = np.zeros((65, 12), dtype=np.uint8)
    lamps[18:34, 0] = 1
    travel = lamps.astype(float) * .002
    force = lamps.astype(float) * .5
    physics = dict(q_actual=actual, q_command=q, gripper_actual=np.tile([.004, -.004], (65, 1)),
                   button_travel=travel, contact_force=force, lights=lamps)
    idx = np.arange(17) * 4
    future = np.minimum(idx + 4, 64)
    frames = dict(state=np.array([pose(kin, v) for v in actual[idx]]),
                  action=np.array([pose(kin, v) for v in q[future]]), sim_time=np.arange(17)/30,
                  q_actual=actual[idx].copy(), q_target=q[future].copy(), physics_index=idx,
                  phase=np.array(["settle"]*2+["approach"]*2+["press"]*6+["retract"]*4+["return_home"]*3), lights=lamps[idx].copy())
    collection = dict(raw_schema_version=10, fps=30, physics_hz=120, capture_stride=4, action_horizon_s=1/30,
                      config=CFG, collection_fingerprint="unit_fixture", pose_frame="base_link", pose_link="gripper_tcp",
                      tcp_offset_link6_m=[0, 0, .1358], gripper_width_m=.008)
    meta = {k:v for k,v in collection.items() if k != "config"}
    meta.update(episode_id=0, floor=24, task="Press 24 floor.", success=True, unexpected_collisions=[],
                num_frames=17, physics_steps=65, plan_physics_steps=65,
                events=[dict(type="pressed", floor=24, physics_index=18, time=18/120),
                        dict(type="released", floor=24, physics_index=34, time=34/120)])
    return frames, physics, meta, collection


def inspect(source):
    return inspect_press_prefix_arrays(*source, project_root=ROOT)


def test_terminal_copies_last_executed_planned_target_not_future_or_state(source):
    frames, physics, _, _ = source
    originals = copy.deepcopy(source)
    cut = inspect(source)
    assert cut["cut_frame_index"] == 5 and cut["first_press_physics_index"] == 18
    assert cut["kept_frames"] == 6 and cut["kept_physics_steps"] == 21
    assert cut["end_time_s"] == 5/30 and cut["capture_delay_from_first_press_s"] == 2/120
    kept, actual = crop_press_prefix_arrays(frames, physics, cut)
    np.testing.assert_array_equal(kept["action"][:-1], frames["action"][:5])
    np.testing.assert_array_equal(kept["action"][-1], frames["action"][4])
    np.testing.assert_array_equal(kept["action"][-1], kept["action"][-2])
    assert not np.array_equal(kept["action"][-1], frames["action"][5])
    assert not np.array_equal(kept["action"][-1], frames["state"][5])
    np.testing.assert_array_equal(kept["q_target"][-1], physics["q_command"][20])
    for key in frames:
        np.testing.assert_array_equal(frames[key], originals[0][key])
        if key not in ("action", "q_target"):
            np.testing.assert_array_equal(kept[key], frames[key][:6])
        assert not np.shares_memory(kept[key], frames[key])
    for key in physics:
        np.testing.assert_array_equal(actual[key], physics[key][:21])
        assert not np.shares_memory(actual[key], physics[key])
    assert kept["lights"][:, 0].tolist() == [0, 0, 0, 0, 0, 1]


def test_batched_fk_matches_independent_urdf_kinematics(source):
    q = source[1]["q_command"][::7]
    expected = np.stack([PiperKinematics(URDF, .1358).fk(v) for v in q])
    np.testing.assert_allclose(tcp_transforms(q, URDF), expected, atol=1e-14)


def test_schema11_cut_preserves_layout_without_changing_pose_or_terminal_policy(source):
    from test_panel_metadata import panel_fixture
    from pressb.panel_metadata import episode_panel_context
    baseline = inspect(source)
    frames, physics, meta, collection = copy.deepcopy(source)
    panel_collection, panel_meta = panel_fixture()
    collection["raw_schema_version"] = meta["raw_schema_version"] = 11
    collection["config"].update({key: value for key, value in panel_collection["config"].items()
                                 if key.startswith("panel_")})
    for key in ("panel_offset_x_m", "panel_offset_y_m", "panel_layout", "variation", "env_offset_m"):
        meta[key] = panel_meta[key]
    cut = inspect((frames, physics, meta, collection))
    assert episode_panel_context(cut) == episode_panel_context(meta)
    assert {key: cut[key] for key in baseline} == baseline
    kept, _ = crop_press_prefix_arrays(frames, physics, cut)
    assert kept["state"].shape[1] == kept["action"].shape[1] == 8
    np.testing.assert_array_equal(kept["action"][-1], frames["action"][baseline["cut_frame_index"] - 1])


@pytest.mark.parametrize("kind", ["missing_light", "initial_light", "wrong_button", "missing_force", "missing_travel",
                                  "transient_between_frames", "release_before_cut", "wrong_event", "wrong_phase",
                                  "prior_retract", "wrong_timing", "wrong_q_horizon", "wrong_tcp", "wrong_rotation", "nonunit_quaternion"])
def test_rejects_inconsistent_cut_evidence(source, kind):
    frames, physics, meta, _ = source
    if kind == "missing_light":
        physics["lights"][:] = 0
    elif kind == "initial_light":
        physics["lights"][0, 0] = 1
    elif kind == "wrong_button":
        physics["lights"][17, 1] = 1
    elif kind == "missing_force":
        physics["contact_force"][18, 0] = 0
    elif kind == "missing_travel":
        physics["button_travel"][18, 0] = .001
    elif kind == "transient_between_frames":
        physics["lights"][2:4, 0] = 1
        physics["button_travel"][2:4, 0] = .002
        physics["contact_force"][2:4, 0] = .5
    elif kind == "release_before_cut":
        physics["button_travel"][19, 0] = 0
    elif kind == "wrong_event":
        meta["events"][0]["physics_index"] = 17
    elif kind == "wrong_phase":
        frames["phase"][5] = "retract"
    elif kind == "prior_retract":
        frames["phase"][2] = "retract"
    elif kind == "wrong_timing":
        frames["sim_time"] *= 3
    elif kind == "wrong_q_horizon":
        frames["q_target"][4] = physics["q_command"][24]
    elif kind == "wrong_tcp":
        frames["action"][4, 0] += .1042
    elif kind == "wrong_rotation":
        frames["action"][4, 3:7] = [1, 0, 0, 0]
    elif kind == "nonunit_quaternion":
        frames["action"][4, 3:7] *= 2
    frames["lights"] = physics["lights"][frames["physics_index"]].copy()
    with pytest.raises(ValueError):
        inspect(source)


def test_crop_refuses_stale_cut_record_and_mutated_targets(source):
    cut = inspect(source)
    stale = dict(cut, cut_frame_index=6)
    with pytest.raises(ValueError, match="Invalid cut record"):
        crop_press_prefix_arrays(source[0], source[1], stale)
    source[0]["action"][4, 0] += .01
    with pytest.raises(ValueError, match="terminal target mismatch"):
        crop_press_prefix_arrays(source[0], source[1], cut)


@pytest.mark.parametrize("press_step", [17, 20])
def test_first_press_interval_accepts_both_valid_endpoints(source, press_step):
    frames, physics, meta, _ = source
    physics["lights"][:34, 0] = 0
    physics["lights"][press_step:34, 0] = 1
    physics["button_travel"] = physics["lights"].astype(float) * .002
    physics["contact_force"] = physics["lights"].astype(float) * .5
    frames["lights"] = physics["lights"][frames["physics_index"]].copy()
    meta["events"][0].update(physics_index=press_step, time=press_step/120)
    record = inspect(source)
    assert record["cut_frame_index"] == 5
    assert record["first_press_physics_index"] == press_step
