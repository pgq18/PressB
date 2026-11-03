"""Training-unit conversion and accepted-prefix contracts for tabletop export."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_table_buttons import (
    FPS, RAW_SCHEMA, RAW_UNITS, TERMINAL_ACTION, discover_episodes, feature_spec,
    load_episode, parse_group, to_pap_units, write_json,
)


def raw_episode(raw, episode_index=0, count=7):
    directory = raw / f"episode_{episode_index:06d}"
    directory.mkdir(parents=True)
    profile, color = parse_group(raw.name)
    metadata = dict(schema_version=RAW_SCHEMA, episode_index=episode_index, group=raw.name,
                    spacing_profile=profile, target_color=color, instruction=f"Press the {color} button.",
                    success=True, fps=FPS, camera_resolution=[640, 480], state_units=RAW_UNITS,
                    action_units=RAW_UNITS, gripper_representation="full_width",
                    terminal_action=TERMINAL_ACTION, seed=episode_index + 1, physics_dt=1 / 120,
                    frame_count=count, duration_s=(count - 1) / FPS,
                    first_success_time_s=(count - 1) / FPS - 1 / 120)
    state = np.zeros((count, 7), dtype=np.float64)
    state[:, 0] = np.arange(count) / 100 + episode_index / 10
    state[:, 6] = .06
    action = state + np.array([.01, .02, .03, -.04, .05, .06, .01])
    action[-1] = action[-2]
    np.savez(directory / "data.npz", sim_time=np.arange(count) / FPS, state=state, action=action)
    np.savez(directory / "physics.npz", time_s=np.arange(4 * (count - 1) + 1) / 120)
    write_json(directory / "metadata.json", metadata)
    write_json(directory / "report.json", {})
    write_json(directory / "config.json", {})
    for name in ("up.mp4", "wrist.mp4"):
        (directory / name).touch()
    return directory, metadata, state, action


def test_unit_conversion_matches_pap_without_half_width_or_state_shift():
    raw = np.array([[np.pi, -np.pi / 2, 0, np.pi / 4, 0, 0, .08]])
    original = raw.copy()
    converted = to_pap_units(raw)
    np.testing.assert_allclose(converted, [[180., -90., 0., 45., 0., 0., 8.]])
    np.testing.assert_array_equal(raw, original)
    assert converted.dtype == np.float32
    assert feature_spec()["action"]["names"][-1] == "gripper.pos"
    assert feature_spec()["observation.images.up"]["shape"] == (480, 640, 3)


@pytest.mark.parametrize("raw", [np.zeros((2, 6)), np.ones(7), np.full((2, 7), np.nan)])
def test_bad_unit_input_is_rejected(raw):
    with pytest.raises(ValueError):
        to_pap_units(raw)


@pytest.mark.parametrize("profile", ["near", "medium", "far"])
@pytest.mark.parametrize("color", ["red", "yellow", "blue"])
def test_instruction_depends_only_on_color(tmp_path, profile, color):
    directory, _, state, action = raw_episode(tmp_path / f"{profile}_{color}")
    episode = load_episode(directory, f"{profile}_{color}")
    assert episode["metadata"]["instruction"] == f"Press the {color} button."
    np.testing.assert_array_equal(episode["state"], to_pap_units(state))
    np.testing.assert_array_equal(episode["action"], to_pap_units(action))


@pytest.mark.parametrize("key,value", [
    ("instruction", "Press the red button in the near configuration."),
    ("target_color", "blue"), ("spacing_profile", "far"), ("fps", 10),
    ("success", False), ("success", 1), ("state_units", ["degree"] * 6 + ["cm"]),
    ("action_units", ["rad"] * 6 + ["cm"]), ("gripper_representation", "half_width"),
    ("terminal_action", "measured_state"), ("duration_s", 1.),
    ("first_success_time_s", .05), ("first_success_time_s", .25),
])
def test_inconsistent_raw_metadata_fails_before_export(tmp_path, key, value):
    directory, metadata, _, _ = raw_episode(tmp_path / "near_red")
    metadata[key] = value
    write_json(directory / "metadata.json", metadata)
    with pytest.raises(ValueError):
        load_episode(directory, "near_red")


def test_terminal_target_cannot_be_replaced_with_terminal_state(tmp_path):
    directory, _, state, action = raw_episode(tmp_path / "near_red")
    action[-1] = state[-1]
    np.savez(directory / "data.npz", sim_time=np.arange(len(state)) / FPS, state=state, action=action)
    with pytest.raises(ValueError, match="terminal action"):
        load_episode(directory, "near_red")


def test_time_alignment_rejects_missing_control_tick(tmp_path):
    directory, _, state, action = raw_episode(tmp_path / "near_red")
    stamps = np.arange(len(state)) / FPS
    stamps[3:] += 1 / FPS
    np.savez(directory / "data.npz", sim_time=stamps, state=state, action=action)
    with pytest.raises(ValueError, match="timestamps"):
        load_episode(directory, "near_red")


def test_committed_discovery_ignores_unpublished_directory_and_rejects_duplicates(tmp_path):
    raw = tmp_path / "near_red"
    raw_episode(raw)
    (raw / ".episode_000001.inprogress").mkdir()
    assert len(discover_episodes(raw, 2)) == 1
    second, metadata, _, _ = raw_episode(raw, 1)
    metadata["seed"] = 1
    write_json(second / "metadata.json", metadata)
    with pytest.raises(ValueError, match="Repeated layout seed"):
        discover_episodes(raw, 2)


def test_discovery_does_not_silently_truncate_or_skip_numbering(tmp_path):
    raw = tmp_path / "near_red"
    raw_episode(raw, 0)
    raw_episode(raw, 2)
    with pytest.raises(ValueError, match="expected 1"):
        discover_episodes(raw, 1)
    with pytest.raises(ValueError, match="consecutive"):
        discover_episodes(raw, 2)


@pytest.mark.skipif(importlib.util.find_spec("lerobot") is None,
                    reason="Official writer/reader integration runs in the separate LeRobot environment")
def test_official_writer_reader_preserves_two_episode_boundaries_and_units(tmp_path):
    import av
    from export_table_buttons import configure_runtime, export_group, read_json
    configure_runtime()
    raw = tmp_path / "raw" / "medium_blue"
    for episode_index in range(2):
        directory, metadata, _, _ = raw_episode(raw, episode_index, count=9)
        for view_index, name in enumerate(("up.mp4", "wrist.mp4")):
            with av.open(str(directory / name), "w") as container:
                stream = container.add_stream("libx264", rate=FPS)
                stream.width, stream.height, stream.pix_fmt = 640, 480, "yuv420p"
                stream.options = {"crf": "18", "preset": "ultrafast", "g": "2"}
                for index in range(metadata["frame_count"]):
                    image = np.zeros((480, 640, 3), dtype=np.uint8)
                    image[:] = [20 + 60 * episode_index, 25 + index * 10, 40 + view_index * 80]
                    image[100:200, 100 + index * 15:200 + index * 15] = [210, 180, 20]
                    frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
    entries = discover_episodes(raw, 2)
    output = tmp_path / "export" / "medium_blue"
    output.parent.mkdir()
    export_group(raw, output, entries)
    audit = read_json(output / "meta/audit.json")
    assert audit["success"] and audit["full_video_decode"]
    assert audit["frames"] == 18 and audit["episodes"] == 2
    assert audit["official_reader_samples"] == 6
    assert set(audit["decoded_frames"].values()) == {18}
    before = deepcopy(read_json(output / "meta/export_manifest.json"))
    export_group(raw, output, entries)
    assert read_json(output / "meta/export_manifest.json") == before
