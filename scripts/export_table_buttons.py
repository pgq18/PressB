#!/usr/bin/env python3
"""Export one tabletop-button group using the official LeRobot v3 writer.

Run in the separate CPU LeRobot environment. Captures stay untouched; a dataset
appears at --output only after frame, action, language and reader checks pass.
No Hub access or upload is performed. Use --watch while collection is running.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pressb.table_buttons.task import COLORS, task_instruction

FPS = 30
LEROBOT_VERSION = "0.6.1"
RAW_SCHEMA = "pressb.table_buttons.raw.v1"
IMAGE_SHAPE = (480, 640, 3)
NAMES = [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"]
CAMERAS = {"observation.images.up": "up.mp4", "observation.images.wrist": "wrist.mp4"}
PROFILES = ("near", "medium", "far")
RAW_UNITS = ["rad"] * 6 + ["m"]
TERMINAL_ACTION = "repeat_previous_interval_target"


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def configure_runtime():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/hf"))
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    actual = version("lerobot")
    if actual != LEROBOT_VERSION:
        raise RuntimeError(f"Use lerobot=={LEROBOT_VERSION} in its separate CPU environment, found {actual}")
    import av
    import torch
    av.logging.set_level(av.logging.ERROR)
    torch.set_num_threads(2)


def feature_spec():
    """Match the PAP training interface: six degrees and full gripper width cm."""
    return {
        **{key: {"dtype": "float32", "shape": (7,), "names": NAMES.copy()}
           for key in ("action", "observation.state")},
        **{key: {"dtype": "video", "shape": IMAGE_SHAPE,
                 "names": ["height", "width", "channels"]} for key in CAMERAS},
    }


def to_pap_units(raw_values):
    values = np.asarray(raw_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or not np.isfinite(values).all():
        raise ValueError("Expected finite N x 7 radians/full-gripper-metres data")
    converted = values.copy()
    converted[:, :6] = np.rad2deg(values[:, :6])
    converted[:, 6] = values[:, 6] * 100.0
    return converted.astype(np.float32)


def parse_group(name: str):
    match = re.fullmatch(r"(near|medium|far)_(red|yellow|blue)", name)
    if match is None:
        raise ValueError(f"Expected a spacing/color group, found {name!r}")
    return match.groups()


def load_episode(directory: Path, group: str) -> dict:
    """Validate the collector's committed prefix before any unit conversion."""
    profile, color = parse_group(group)
    match = re.fullmatch(r"episode_([0-9]{6,})", directory.name)
    if match is None:
        raise ValueError(f"Invalid committed episode directory: {directory}")
    metadata = read_json(directory / "metadata.json")
    expected = dict(schema_version=RAW_SCHEMA, episode_index=int(match[1]), group=group,
                    spacing_profile=profile, target_color=color, instruction=task_instruction(color),
                    success=True, fps=FPS, camera_resolution=[640, 480], state_units=RAW_UNITS,
                    action_units=RAW_UNITS, gripper_representation="full_width",
                    terminal_action=TERMINAL_ACTION)
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"{directory}: invalid {key}: {metadata.get(key)!r}; expected {value!r}")
    if (type(metadata.get("success")) is not bool or type(metadata.get("seed")) is not int
            or type(metadata.get("frame_count")) is not int
            or abs(float(metadata.get("physics_dt", 0)) - 1 / 120) > 1e-10):
        raise ValueError(f"{directory}: invalid success, seed, frame count or physics timing")
    for name in ("data.npz", "physics.npz", "report.json", "config.json", *CAMERAS.values()):
        if not (directory / name).is_file():
            raise ValueError(f"Incomplete committed episode: {directory / name}")
    with np.load(directory / "data.npz", allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in ("sim_time", "state", "action")}
    count = metadata["frame_count"]
    if count < 2 or arrays["sim_time"].shape != (count,):
        raise ValueError(f"{directory}: invalid frame count or sim_time shape")
    if not np.allclose(arrays["sim_time"], np.arange(count) / FPS, atol=5e-6, rtol=0):
        raise ValueError(f"{directory}: timestamps must start at zero and advance at {FPS} Hz")
    for key in ("state", "action"):
        if arrays[key].shape != (count, 7) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"{directory}: {key} must contain {count} finite 7D samples")
    if not np.array_equal(arrays["action"][-1], arrays["action"][-2]):
        raise ValueError(f"{directory}: terminal action must repeat the previous interval target")
    duration = float(metadata["duration_s"])
    success_time = float(metadata["first_success_time_s"])
    if (not np.isfinite([duration, success_time]).all() or success_time <= 0
            or abs(duration - (count - 1) / FPS) > 5e-6
            or duration < success_time - 5e-6 or duration - success_time > 1 / FPS + 5e-6):
        raise ValueError(f"{directory}: episode must end at the first sampled success boundary")
    return {"directory": directory, "metadata": metadata, "frames": count,
            "state": to_pap_units(arrays["state"]), "action": to_pap_units(arrays["action"])}


def discover_episodes(raw: Path, expected_count: int) -> list[dict]:
    parse_group(raw.name)
    directories = sorted(path for path in raw.glob("episode_*")
                         if path.is_dir() and re.fullmatch(r"episode_[0-9]{6,}", path.name))
    if len(directories) > expected_count:
        raise ValueError(f"Found {len(directories)} episodes, expected {expected_count}; do not silently truncate")
    episodes = [load_episode(path, raw.name) for path in directories]
    if [entry["metadata"]["episode_index"] for entry in episodes] != list(range(len(episodes))):
        raise ValueError("Committed episode indices must be consecutive, starting at zero")
    seeds = [entry["metadata"]["seed"] for entry in episodes]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Repeated layout seed in the same dataset")
    return episodes


def video_frames(path: Path):
    import av
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1 or container.streams.audio:
            raise ValueError(f"Expected one video stream without audio: {path}")
        stream = container.streams.video[0]
        stream.thread_count = 1
        if (stream.width, stream.height) != (640, 480) or abs(float(stream.average_rate) - FPS) > 1e-8:
            raise ValueError(f"Expected 640 x 480 at {FPS} Hz: {path}")
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None or abs(float(frame.pts * frame.time_base) - index / FPS) > 5e-6:
                raise ValueError(f"Nonconsecutive video timestamp at {path}, frame {index}")
            yield frame.to_ndarray(format="rgb24")


def selected_video_frames(path: Path, selection: set[int]):
    return {index: image for index, image in enumerate(video_frames(path)) if index in selection}


def manifest_for(raw: Path, episodes: list[dict]) -> dict:
    profile, color = parse_group(raw.name)
    return dict(schema_version=1, format_version="v3.0", lerobot_version=LEROBOT_VERSION,
                group=raw.name, spacing_profile=profile, target_color=color,
                instruction=task_instruction(color), raw_root=str(raw), fps=FPS,
                episodes=[entry["metadata"] for entry in episodes],
                semantics=dict(state="measured_joint_position_and_full_gripper_width",
                               action="commanded_target_for_following_1_over_30_second_interval",
                               units=["degree"] * 6 + ["cm"], joint_names=NAMES,
                               terminal_action=TERMINAL_ACTION,
                               episode_end="first_30Hz_observation_after_measured_target_press_success",
                               image_layout="RGB HWC 480 x 640; observation.images.up and .wrist"),
                encoding=dict(codec="h264", crf=18, preset="ultrafast", g=2,
                              writer="official LeRobot create/add_frame/save_episode",
                              source_mp4_reencoded=True, frame_dropping=False))


def audit_dataset(root: Path, episodes: list[dict]) -> dict:
    """Read with the official loader and independently compare every numeric row."""
    import pandas as pd
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    manifest = read_json(root / "meta/export_manifest.json")
    info = read_json(root / "meta/info.json")
    total = sum(entry["frames"] for entry in episodes)
    if (info["codebase_version"] != "v3.0" or info["fps"] != FPS
            or info["total_episodes"] != len(episodes) or info["total_frames"] != total
            or info["total_tasks"] != 1):
        raise ValueError("Exported info.json disagrees with source episode/frame/task counts")
    for key, spec in feature_spec().items():
        if any(info["features"][key][field] != (list(value) if isinstance(value, tuple) else value)
               for field, value in spec.items()):
            raise ValueError(f"Exported feature differs from PAP interface: {key}")
    tasks = pd.read_parquet(root / "meta/tasks.parquet")
    if tasks.index.tolist() != [manifest["instruction"]] or tasks["task_index"].tolist() != [0]:
        raise ValueError("Exported task vocabulary changed")
    table = pd.concat([pd.read_parquet(path) for path in sorted((root / "data").rglob("*.parquet"))],
                      ignore_index=True).sort_values("index")
    if len(table) != total or table["index"].tolist() != list(range(total)):
        raise ValueError("Exported global frame index has gaps or duplicates")
    dataset = LeRobotDataset(f"local/{manifest['group']}", root=root, video_backend="pyav")
    max_mae, offset, sample_count = 0., 0, 0
    for episode_index, entry in enumerate(episodes):
        count = entry["frames"]
        rows = table.iloc[offset:offset + count]
        if (rows["episode_index"].tolist() != [episode_index] * count
                or rows["frame_index"].tolist() != list(range(count))
                or rows["task_index"].tolist() != [0] * count
                or not np.allclose(rows["timestamp"], np.arange(count) / FPS, atol=5e-6, rtol=0)):
            raise ValueError(f"Invalid episode/time/task indices: {episode_index}")
        for feature, source_key in (("observation.state", "state"), ("action", "action")):
            if not np.array_equal(np.stack(rows[feature]), entry[source_key]):
                raise ValueError(f"State/action conversion or alignment mismatch: {episode_index}, {feature}")
        selection = {0, count // 2, count - 1}
        sources = {key: selected_video_frames(entry["directory"] / name, selection)
                   for key, name in CAMERAS.items()}
        for local_index in sorted(selection):
            sample = dataset[offset + local_index]
            if sample["task"] != manifest["instruction"]:
                raise ValueError(f"Official reader returned the wrong instruction: {episode_index}")
            for key in CAMERAS:
                decoded = sample[key].cpu().numpy()
                if decoded.shape != (3, 480, 640) or not np.isfinite(decoded).all():
                    raise ValueError(f"Official reader returned malformed RGB: {key}")
                decoded = np.moveaxis(decoded, 0, -1) * 255.0
                mae = float(np.abs(decoded - sources[key][local_index].astype(float)).mean())
                if mae > 8.0:
                    raise ValueError(f"Camera/frame correspondence failed: episode {episode_index}, {key}, {mae=}")
                max_mae = max(mae, max_mae)
            sample_count += 1
        offset += count
    decoded_counts = {}
    for key in CAMERAS:
        count = sum(sum(1 for _ in video_frames(path)) for path in sorted((root / "videos" / key).rglob("*.mp4")))
        if count != total:
            raise ValueError(f"Output video dropped/duplicated frames: {key}, {count} vs {total}")
        decoded_counts[key] = count
    return dict(success=True, episodes=len(episodes), frames=total, full_video_decode=True,
                decoded_frames=decoded_counts, official_reader_samples=sample_count,
                max_source_to_output_rgb_mae=max_mae, numeric_rows_compared=total,
                checked_utc=datetime.now(timezone.utc).isoformat())


def write_readme(root: Path, manifest: dict, audit: dict):
    text = f"""# PiPER tabletop buttons: {manifest['group']}

{audit['episodes']} successful Isaac Sim demonstrations, {audit['frames']} frames at 30 Hz.
Instruction for every episode: **{manifest['instruction']}**

Each spacing group retains exactly the same instruction for the same color.
Only successful measured button presses are exported. Episodes include the initial
observation and end at the first 30 Hz observation following success; retraction
is excluded. Layout seeds and all three button positions are preserved per episode
in `meta/export_manifest.json`.

`action` and `observation.state` contain six joint angles in degrees followed by
the **full gripper opening in centimetres**, matching the real PAP dataset's 7D
interface. State is measured; action is the commanded target applied during the
following 1/30 s interval. The terminal action repeats the previous target.
Two RGB cameras are stored at 640 x 480 and 30 Hz: `observation.images.up` and
`observation.images.wrist`. The wrist camera uses the tabletop-button/elevator
configuration, not the PAP calibration.

Written with official LeRobot {LEROBOT_VERSION} (`v3.0`) APIs. Source H264 videos
are decoded and re-encoded in sequence; no source frames are dropped or retimed.
`meta/audit.json` records full video decoding, comparison of every numeric row,
and first/middle/last RGB samples through the official reader for every episode.
The original raw collection is at `{manifest['raw_root']}`. No Hub upload was made.

```python
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("local/{manifest['group']}", root=Path(__file__).parent,
                         video_backend="pyav")
sample = dataset[0]
```
"""
    (root / "README.md").write_text(text)


@contextmanager
def output_lock(output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / f".{output.name}.export.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"An exporter already owns {output}") from error
        yield


def export_group(raw: Path, output: Path, episodes: list[dict]):
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    manifest = manifest_for(raw, episodes)
    if output.exists():
        if read_json(output / "meta/export_manifest.json") != manifest:
            raise ValueError(f"Existing output belongs to different data/settings: {output}")
        audit_dataset(output, episodes)
        print(f"Already exported and verified: {output}", flush=True)
        return
    staging = output.with_name(f".{output.name}.inprogress-{uuid.uuid4().hex[:8]}")
    queue_capacity = max(entry["frames"] for entry in episodes) + 1
    dataset = LeRobotDataset.create(
        repo_id=f"local/{raw.name}", root=staging, fps=FPS, robot_type="piper_follower",
        features=feature_spec(), use_videos=True, video_backend="pyav",
        rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", crf=18, g=2, preset="ultrafast"),
        streaming_encoding=True, encoder_threads=2, encoder_queue_maxsize=queue_capacity,
        metadata_buffer_size=1,
    )
    try:
        for entry in episodes:
            directory, count = entry["directory"], entry["frames"]
            generators = {key: video_frames(directory / name) for key, name in CAMERAS.items()}
            try:
                for index in range(count):
                    try:
                        images = {key: next(reader) for key, reader in generators.items()}
                    except StopIteration as error:
                        raise ValueError(f"Short source video in {directory}, frame {index}") from error
                    dataset.add_frame({**images, "observation.state": entry["state"][index],
                                       "action": entry["action"][index], "task": manifest["instruction"]})
                for key, reader in generators.items():
                    if next(reader, None) is not None:
                        raise ValueError(f"Video exceeds sampled prefix: {directory}, {key}")
                dataset.save_episode()
            finally:
                for reader in generators.values():
                    reader.close()
            print(f"Exported {raw.name} {entry['metadata']['episode_index'] + 1}/{len(episodes)}: {count} frames", flush=True)
    finally:
        dataset.finalize()
    write_json(staging / "meta/export_manifest.json", manifest)
    for name in ("collection_metadata.json", "plan.json"):
        if (raw / name).is_file():
            shutil.copy2(raw / name, staging / "meta" / name)
    audit = audit_dataset(staging, episodes)
    write_json(staging / "meta/audit.json", audit)
    write_readme(staging, manifest, audit)
    staging.rename(output)
    print(f"Published {len(episodes)} verified episodes: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path, help="One group, e.g. raw/near_red")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--watch", action="store_true", help="Wait for all --episodes successful captures")
    parser.add_argument("--poll-seconds", type=float, default=15)
    args = parser.parse_args()
    if args.episodes < 1 or not np.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        parser.error("Episode count and polling interval must be positive")
    raw, output = args.raw.resolve(), args.output.resolve()
    parse_group(raw.name)
    if raw == output or raw in output.parents or output in raw.parents:
        parser.error("Raw and output must be separate directories")
    configure_runtime()
    with output_lock(output):
        last_count = None
        while True:
            episodes = discover_episodes(raw, args.episodes)
            if len(episodes) == args.episodes:
                export_group(raw, output, episodes)
                return
            if not args.watch:
                raise ValueError(f"Only {len(episodes)}/{args.episodes} successful committed episodes; use --watch")
            if len(episodes) != last_count:
                print(f"Waiting for {raw.name}: {len(episodes)}/{args.episodes} committed episodes", flush=True)
                last_count = len(episodes)
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
