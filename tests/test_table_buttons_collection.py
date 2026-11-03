"""Catch wrong labels, action time shifts, hidden presses and malformed videos."""
from fractions import Fraction
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from pressb.table_buttons.collection import measured_state, validate_raw_arrays

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("table_buttons_dataset_audit", ROOT / "scripts/audit_table_buttons_dataset.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def records():
    action = np.zeros((3, 7))
    action[:, 6] = .0002
    action[0, 0], action[1:, 0] = .01, .02
    q = np.zeros((9, 8))
    q[:, 6], q[:, 7] = .0001, -.0001
    q[1:5, 0], q[5:, 0] = .01, .02
    physics = dict(time_s=np.arange(9) / 120, q_actual=q.copy(), q_target=q.copy(),
                   travel_m=np.zeros((9, 3)), finger_force_n=np.zeros((9, 3)))
    physics["travel_m"][4:, 0] = .0032
    physics["finger_force_n"][4:, 0] = 1.
    frames = dict(sim_time=np.arange(3) / 30,
                  state=np.column_stack((q[::4, :6], q[::4, 6] - q[::4, 7])), action=action)
    contacts = [dict(tick=tick, color="red", fingers=["link7"], force_n=1.) for tick in range(4, 9)]
    return frames, physics, contacts


def config():
    cfg = json.loads((ROOT / "configs/piper_table_buttons.json").read_text())
    cfg["initial_joint_positions_rad"] = [0.] * 6
    cfg["position_sampling"]["spacing_profile"] = "near"
    positions = dict(red=[.33, 0.], yellow=[.40, 0.], blue=[.365, .06])
    for color, xy in positions.items():
        cfg["buttons"][color]["center_xy"] = xy
    return cfg


def test_actual_actions_and_terminal_are_aligned_independently():
    frames, physics, _ = records()
    validate_raw_arrays(frames, physics)
    assert audit.audit_arrays(frames, physics) == (3, 9)
    assert measured_state(physics["q_actual"][0])[6] == pytest.approx(.0002)


@pytest.mark.parametrize("mutation,match", [
    ("shifted_frame_clock", "timestamp|Frame time"),
    ("gripper_half_width", "states|State/RGB"),
    ("future_action", "target|Action"),
    ("radians_as_degrees", "target|Action"),
    ("terminal_new_action", "Terminal"),
    ("missing_physics_tick", "timelines|endpoints"),
])
def test_frame_action_corruption_is_rejected(mutation, match):
    frames, physics, _ = records()
    if mutation == "shifted_frame_clock":
        frames["sim_time"] += 1 / 120
    elif mutation == "gripper_half_width":
        frames["state"][:, 6] /= 2
    elif mutation == "future_action":
        frames["action"][0] = frames["action"][1]
    elif mutation == "radians_as_degrees":
        frames["action"][:, :6] = np.rad2deg(frames["action"][:, :6])
    elif mutation == "terminal_new_action":
        frames["action"][-1, 1] = .5
    else:
        physics = {key: value[:-1] for key, value in physics.items()}
    for validator in (validate_raw_arrays, audit.audit_arrays):
        with pytest.raises(ValueError, match=match):
            validator(frames, physics)


def test_physical_press_reconstructed_from_dwell_and_contact_events():
    _, physics, contacts = records()
    report = audit.audit_physics(physics, config(), "red", contacts, [])
    assert report["first_success_time_s"] == pytest.approx(7 / 120)
    assert report["press_events"] == [dict(time_s=7 / 120, color="red")]


@pytest.mark.parametrize("corruption,match", [
    ("interrupted_force", "does not yield"),
    ("missing_contact", "events and per-tick"),
    ("wrong_after_success", "Non-target finger contact"),
    ("table_collision", "Unexpected"),
])
def test_physical_failures_cannot_pass_using_latched_success(corruption, match):
    _, physics, contacts = records()
    unexpected = []
    if corruption == "interrupted_force":
        physics["finger_force_n"][6, 0] = 0
    elif corruption == "missing_contact":
        contacts.pop()
    elif corruption == "wrong_after_success":
        physics["finger_force_n"][8, 1] = 1.
        contacts.append(dict(tick=8, color="yellow", fingers=["link8"], force_n=1.))
    else:
        unexpected.append(dict(tick=2, actors=["/World/Piper/link2", "/World/Table/Top"], force_n=.3))
    with pytest.raises(ValueError, match=match):
        audit.audit_physics(physics, config(), "red", contacts, unexpected)


def create_episode(tmp_path):
    directory = tmp_path / "near_red" / "episode_000000"
    directory.mkdir(parents=True)
    frames, physics, contacts = records()
    cfg = config()
    metadata = dict(schema_version="pressb.table_buttons.raw.v1", episode_index=0, group="near_red",
                    spacing_profile="near", target_color="red", instruction="Press the red button.",
                    seed=0, fps=30, physics_dt=1 / 120, camera_resolution=[640, 480],
                    state_units=["rad"] * 6 + ["m"], action_units=["rad"] * 6 + ["m"],
                    gripper_representation="full_width", positions_xy_m={color: value["center_xy"] for color, value in cfg["buttons"].items()},
                    success=True, first_success_time_s=7 / 120, duration_s=2 / 30,
                    frame_count=3, terminal_action="repeat_previous_interval_target")
    for name, value in (("metadata", metadata), ("config", cfg), ("contacts", contacts), ("unexpected_contacts", [])):
        (directory / f"{name}.json").write_text(json.dumps(value))
    np.savez_compressed(directory / "data.npz", **frames)
    np.savez_compressed(directory / "physics.npz", **physics)
    return directory


def test_episode_checks_physical_success_and_spacing(tmp_path):
    directory = create_episode(tmp_path)
    report = audit.audit_episode(directory, "near_red", decode_videos=False)
    assert report["success"] and report["frames"] == 3
    metadata = audit.read_json(directory / "metadata.json")
    metadata["instruction"] = "Press the near red button."
    (directory / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="instruction"):
        audit.audit_episode(directory, "near_red", decode_videos=False)


def test_rejected_episodes_are_never_counted_as_accepted(tmp_path):
    create_episode(tmp_path)
    rejected = tmp_path / "rejected" / "near_red" / "seed_123"
    rejected.mkdir(parents=True)
    (rejected / "metadata.json").write_text('{"success": false}')
    report = audit.audit_collection(tmp_path, 1, groups=["near_red"], decode_videos=False)
    assert report["success"] and report["total_episodes"] == 1
    report = audit.audit_collection(tmp_path, 2, groups=["near_red"], decode_videos=False)
    assert not report["success"]


def write_video(path, count=3, pts_offset=0):
    av = pytest.importorskip("av")
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:, :320] = (20, 120, 220)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 640, 480, "yuv420p"
        stream.options = {"preset": "ultrafast", "crf": "20", "threads": "1"}
        for index in range(count):
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts, frame.time_base = index + pts_offset, Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_video_audit_detects_missing_frame_and_start_offset(tmp_path):
    path = tmp_path / "video.mp4"
    write_video(path)
    assert audit.audit_video(path, 3)["full_decode"]
    with pytest.raises(ValueError, match="count mismatch"):
        audit.audit_video(path, 4)
    shifted = tmp_path / "shifted.mp4"
    write_video(shifted, pts_offset=1)
    with pytest.raises(ValueError, match="PTS"):
        audit.audit_video(shifted, 3)
