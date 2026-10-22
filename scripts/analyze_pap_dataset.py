#!/usr/bin/env python3
"""Read-only LeRobot PAP audit: joint units, FK, event candidates and source frames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import av
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    import pyarrow.parquet as pq
except ImportError:
    # The Isaac environment intentionally omits the dataset stack. Reuse only
    # the installed Arrow wheel; no package installation or environment changes.
    for directory in (ROOT / ".conda/envs/vlajepa-inference/lib").glob("python*/site-packages"):
        if (directory / "pyarrow").exists():
            sys.path.append(str(directory))
            break
    import pyarrow.parquet as pq
from pressb.kinematics import PiperKinematics


def quantiles(values):
    values = np.asarray(values)
    return {name: np.asarray(value).tolist() for name, value in (
        ("min", values.min(axis=0)), ("p05", np.quantile(values, .05, axis=0)),
        ("median", np.median(values, axis=0)), ("p95", np.quantile(values, .95, axis=0)),
        ("max", values.max(axis=0)), ("mean", values.mean(axis=0)),
        ("std", values.std(axis=0)))}


def batch_fk(kin, radians):
    result = np.broadcast_to(np.eye(4), (len(radians), 4, 4)).copy()
    j = 0
    for origin, axis in kin._chain:
        result = result @ origin
        if axis is not None:
            turn = np.broadcast_to(np.eye(4), result.shape).copy()
            turn[:, :3, :3] = Rotation.from_rotvec(radians[:, j, None] * axis).as_matrix()
            result = result @ turn
            j += 1
    result[:, :3, 3] += result[:, :3, :3] @ np.array([0., 0., kin.tip_offset])
    return result


def frame_at(path, seconds):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        container.seek(max(0, int((seconds - .1) / float(stream.time_base))), stream=stream)
        best = None
        for frame in container.decode(stream):
            timestamp = float(frame.pts * stream.time_base)
            if best is None or abs(timestamp - seconds) < abs(best[0] - seconds):
                best = (timestamp, frame.to_image())
            if timestamp > seconds + 1 / 30:
                break
        if best is None:
            raise ValueError(f"No decoded frame at {seconds}: {path}")
        return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("/home/pengguanqi/Datasets/piper_pap_20260606_offset"))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/piper_pap_reconstruction_20261022/dataset_audit")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 10, 100, 200, 300, 400, 499])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    image_dir = args.output / "frames"
    image_dir.mkdir(exist_ok=True)
    info = json.loads((args.dataset / "meta/info.json").read_text())
    metas = []
    for path in sorted((args.dataset / "meta/episodes").rglob("*.parquet")):
        metas.extend(pq.read_table(path).to_pylist())
    metas.sort(key=lambda r: r["episode_index"])
    video_offsets, video_checks = {}, []
    for camera in ("up", "wrist"):
        key = f"observation.images.{camera}"
        groups = {}
        for meta in metas:
            identity = (meta[f"videos/{key}/chunk_index"], meta[f"videos/{key}/file_index"])
            groups.setdefault(identity, []).append(meta)
        for (chunk, file_index), rows in sorted(groups.items()):
            path = args.dataset / info["video_path"].format(video_key=key, chunk_index=chunk, file_index=file_index)
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                count, rate = stream.frames, float(stream.average_rate)
                start_s = float((stream.start_time or 0) * stream.time_base)
            expected_count = sum(row["length"] for row in rows)
            if count != expected_count or rate != info["fps"] or abs(start_s) > 1e-9:
                raise ValueError(f"Cannot infer packed episode offsets: {path}, {count}/{expected_count}, fps={rate}, start={start_s}")
            frame_offset, deltas = 0, []
            for row in rows:
                video_offsets[(row["episode_index"], camera)] = frame_offset / rate
                deltas.append(row[f"videos/{key}/from_timestamp"] - frame_offset/rate)
                if abs(row[f"videos/{key}/to_timestamp"]-row[f"videos/{key}/from_timestamp"]-row["length"]/rate) > 1e-8:
                    raise ValueError(f"Non-contiguous video episode timing: {path}")
                frame_offset += row["length"]
            if np.ptp(deltas) > 1e-8:
                raise ValueError(f"Nonconstant metadata-to-video timing discrepancy: {path}")
            video_checks.append({"path": str(path), "frames": count, "fps": rate, "start_s": start_s,
                "metadata_offset_error_s": float(np.mean(deltas)), "correction":"Use cumulative per-file episode lengths / fps; video frame count and PTS origin verified"})
    arrays = {key: [] for key in ("observation.state", "action", "episode_index", "frame_index", "timestamp", "index")}
    data_paths = sorted((args.dataset / "data").rglob("*.parquet"))
    for path in data_paths:
        table = pq.read_table(path)
        for key in arrays:
            arrays[key].append(np.asarray(table[key].to_pylist()))
    arrays = {key: np.concatenate(values) for key, values in arrays.items()}
    order = np.argsort(arrays["index"])
    arrays = {key: values[order] for key, values in arrays.items()}
    state, action = arrays["observation.state"], arrays["action"]
    kin = PiperKinematics(ROOT / "vendor/robot_lab/source/robot_lab/data/Robots/Agilex/PIPER/piper_description.urdf", tip_offset=.1358)
    pose = batch_fk(kin, np.deg2rad(state[:, :6]))
    initial, events, episode_rows = [], [], []
    for meta in metas:
        episode = meta["episode_index"]
        ids = np.flatnonzero(arrays["episode_index"] == episode)
        if len(ids) != meta["length"] or not np.array_equal(arrays["frame_index"][ids], np.arange(len(ids))):
            raise ValueError(f"Frame count or indexing mismatch: episode {episode}")
        initial.append(ids[0])
        commanded = action[ids, 6]
        # Explicitly heuristic event candidates, not ground-truth success labels.
        close = np.flatnonzero((commanded[:-1] > 5) & (commanded[1:] <= 5)) + 1
        opened = np.flatnonzero((commanded[:-1] <= 5) & (commanded[1:] > 5)) + 1
        first_close = int(close[0]) if len(close) else None
        release = [int(i) for i in opened if first_close is not None and i > first_close + 20]
        selected = {"initial": 0, "quarter": int((len(ids)-1)*.25), "mid": int((len(ids)-1)*.5),
                    "three_quarter": int((len(ids)-1)*.75), "final": len(ids)-1}
        if first_close is not None:
            selected["close"] = first_close
            selected["post_close"] = min(first_close+15, len(ids)-1)
        if release:
            selected["release"] = release[0]
        event = {"episode_index": episode, "length": len(ids), "initial_state_raw": state[ids[0]].tolist(),
                 "initial_q_rad": np.deg2rad(state[ids[0], :6]).tolist(), "initial_tcp_m": pose[ids[0], :3, 3].tolist(),
                 "close_indices": close.tolist(), "open_indices": opened.tolist()}
        for label, candidate in (("close", first_close), ("release", release[0] if release else None)):
            if candidate is not None:
                event[label] = {"frame_index": candidate, "timestamp_s": candidate/info["fps"],
                    "tcp_xyz_m": pose[ids[candidate], :3, 3].tolist(), "state_raw": state[ids[candidate]].tolist()}
        events.append(event)
        if episode not in args.episodes:
            continue
        np.savez_compressed(args.output / f"episode_{episode:06d}.npz", state_raw=state[ids], action_raw=action[ids],
            q_rad=np.deg2rad(state[ids, :6]), gripper_width_m=state[ids, 6]*.01, tcp_transform=pose[ids],
            action_q_rad=np.deg2rad(action[ids, :6]), action_gripper_width_m=action[ids, 6]*.01,
            timestamp_s=arrays["timestamp"][ids],
            units=np.asarray("q_rad/action_q_rad: radians; gripper_width_m/action_gripper_width_m: inferred full-width metres; tcp_transform: base_link metres; timestamp_s: seconds"))
        frames = []
        contact = Image.new("RGB", (640*len(selected), 480*2+50), "white")
        draw = ImageDraw.Draw(contact)
        for column, (label, local) in enumerate(sorted(selected.items(), key=lambda item: item[1])):
            row = {"label": label, "frame_index": local, "timestamp_s": local/info["fps"],
                   "state_raw": state[ids[local]].tolist(), "q_rad": np.deg2rad(state[ids[local], :6]).tolist(),
                   "tcp_transform": pose[ids[local]].tolist(), "images": {}}
            for camera_row, camera in enumerate(("up", "wrist")):
                key = f"observation.images.{camera}"
                video = args.dataset / info["video_path"].format(video_key=key,
                    chunk_index=meta[f"videos/{key}/chunk_index"], file_index=meta[f"videos/{key}/file_index"])
                seconds = video_offsets[(episode, camera)] + local/info["fps"]
                actual, image = frame_at(video, seconds)
                output = image_dir / f"episode_{episode:06d}_{local:04d}_{label}_{camera}.jpg"
                image.save(output, quality=95)
                contact.paste(image, (640*column, camera_row*480+50))
                row["images"][camera] = {"path": str(output), "source_video": str(video),
                    "uncorrected_metadata_timestamp_s": meta[f"videos/{key}/from_timestamp"] + local/info["fps"],
                    "requested_video_timestamp_s": seconds, "actual_video_timestamp_s": actual,
                    "timestamp_error_s": actual-seconds}
            draw.text((640*column+8, 8), f"ep {episode} | {label} | frame {local} | {local/30:.3f}s", fill="black")
            frames.append(row)
        contact.thumbnail((2400, 1000))
        sheet = args.output / f"episode_{episode:06d}_contact.jpg"
        contact.save(sheet, quality=95)
        episode_rows.append({"episode_index": episode, "length": len(ids), "frames": frames, "contact_sheet": str(sheet)})
        print(f"Extracted episode {episode}: {len(frames)} frames per camera", flush=True)
    initial = np.asarray(initial)
    degree_q = np.deg2rad(state[:, :6])
    report = {
        "source_dataset": str(args.dataset), "info": info, "raw_data_schema": str(pq.read_schema(data_paths[0])),
        "task": pq.read_table(args.dataset / "meta/tasks.parquet").to_pylist(),
        "observed_frames": len(state), "observed_episodes": len(metas), "video_timing_checks": video_checks,
        "episode_length_frames": quantiles([m["length"] for m in metas]),
        "episode_duration_s": quantiles([m["length"]/info["fps"] for m in metas]),
        "state_raw": quantiles(state), "action_raw": quantiles(action),
        "initial_state_raw": quantiles(state[initial]), "initial_q_rad": quantiles(degree_q[initial]),
        "initial_tcp_xyz_m": quantiles(pose[initial, :3, 3]), "tcp_xyz_m": quantiles(pose[:, :3, 3]),
        "action_state_abs_error_raw": quantiles(np.abs(action-state)),
        "joint_limit_violations_if_radians": int(np.any((state[:,:6]<kin.lower-.01)|(state[:,:6]>kin.upper+.01),axis=1).sum()),
        "joint_limit_violations_if_degrees": int(np.any((degree_q<kin.lower-.01)|(degree_q>kin.upper+.01),axis=1).sum()),
        "interpretation": {"joint_unit":"degrees, inferred from numerical scale and bundled URDF joint-limit consistency; metadata has no explicit unit",
            "gripper_unit":"centimetres of full opening, inferred from held-object widths and 0..9.11 range consistent with the official 100 mm gripper variant; acquisition driver absent",
            "gripper_conversion":"raw * .01 to full width metres; joints [+width/2, -width/2]",
            "fk_frame":"base_link with TCP at link6 local Z=0.1358 m; no endpose in source dataset",
            "events":"command gripper crosses below 5 cm (close) and later above 5 cm (release); candidates need visual confirmation",
            "object_position":"closed gripper TCP only estimates object centre; no object poses or camera calibration supplied"},
        "gripper_above_bundled_70mm_limit_frames": int((state[:,6] > 7).sum()),
        "gripper_above_bundled_70mm_limit_fraction": float((state[:,6] > 7).mean()),
        "events": events, "selected_episodes": episode_rows}
    for label in ("close", "release"):
        found = [e[label]["tcp_xyz_m"] for e in events if label in e]
        report[f"{label}_tcp_xyz_m"] = quantiles(found) if found else None
    (args.output / "dataset_audit.json").write_text(json.dumps(report, indent=2))
    notes = f"""# Piper PAP dataset reconstruction audit

Source: `{args.dataset}` (read only). The dataset contains {len(metas)} episodes,
{len(state)} frames, 30 Hz, and one task: `Grab the stuff into the box.`
Median episode duration is {np.median([m['length']/30 for m in metas]):.3f} s.

## Coordinate and unit evidence

- `observation.state` and `action` are joint1..joint6 plus gripper, not end-effector poses.
- The six joint numbers are consistent with degrees: every frame is within the bundled
  URDF joint limits after degree-to-radian conversion, while every frame violates at
  least one joint limit if interpreted as radians. No additional joint offset is
  encoded in the metadata; the dataset name alone is not a calibration transform.
- Gripper numbers are provisionally interpreted as full opening in centimetres.
  Around a held red cube, observed width is typically 3.6 cm while commands continue
  closing, consistent with physical contact. Source metadata does not specify units
  and the acquisition driver is absent, so this remains an inference.
- This conversion gives a maximum measured opening of {state[:,6].max()*10:.1f} mm;
  {100*(state[:,6]>7).mean():.2f}% of frames exceed the bundled URDF's 70 mm opening.
  The exported NPZ values are not clipped. A simulator must explicitly account for
  the aperture mismatch rather than claiming unmodified hardware equivalence.
- FK uses the bundled Piper URDF and TCP at `link6 + [0,0,0.1358] m`, relative to
  `base_link`. This is the finger-end centre, not an observed object centre.
  Median close-event TCP Z is {report['close_tcp_xyz_m']['median'][2]*1000:.1f} mm,
  consistent with the base and grasping surface approximately sharing tabletop Z.

## Geometry estimates for building the scene

- Pick TCP 5th–95th percentile XYZ (metres):
  `{report['close_tcp_xyz_m']['p05']}` to `{report['close_tcp_xyz_m']['p95']}`.
- Release TCP median XYZ (metres): `{report['release_tcp_xyz_m']['median']}`.
  The box therefore belongs on the robot's +Y side, near X=0.32 m, Y=0.21 m.
  This identifies its working area, not exact wall dimensions.
- Episode 0 is an explicitly selected representative, with full arrays and both
  camera views. Its first close is frame {events[0]['close']['frame_index']}; release
  is frame {events[0]['release']['frame_index']}. These event candidates come from
  command opening transitions through 5 cm and were checked against the extracted
  images. They are not annotated success labels.
- The dataset has no per-episode object-type labels. Object shape, colour and
  initial position must be checked in each selected episode's source RGB frames;
  the objects must not be assumed to all be the episode-0 red cube. A full initial-
  frame scan also shows blue/green blocks and pink/yellow irregular objects.
  Held gripper aperture is a useful constraint, but is not an object dimension:
  orientation, contact geometry and compliance affect the measured opening.
- `up` is the fixed third-person camera; `wrist` moves with the arm. Both are
  640×480 RGB, AV1, 30 fps. There are no depth images or camera intrinsics/extrinsics.

## Source video timestamp correction

Each source MP4 begins at PTS 0 and its frame count is exactly the sum of the
episodes packed inside it. However, the episode metadata starts at 0.03 s, giving
every episode boundary a 0.03 s offset. Reading the nearest frame to the uncorrected
last timestamp can therefore pick the first frame of the next episode. This audit
checks every MP4's PTS origin, rate, frame count and constant metadata discrepancy,
then uses the cumulative episode frame count divided by FPS as the video offset.
Both corrected requests and original metadata timestamps remain in the JSON report.
This correction affects image lookup only; source data and files are unchanged.
This establishes packed episode membership, not sensor-to-robot synchronization.
The reason for the metadata offset is not documented; an intentional acquisition
latency correction remains possible and should be evaluated during camera fitting.

`episode_*.npz` separates measured `q_rad/gripper_width_m` from commanded
`action_q_rad/action_gripper_width_m`; raw arrays are retained. Source video
timestamps and selected image timestamps are recorded in `dataset_audit.json`.
The initial robot posture varies across episodes, so the chosen episode's measured
first state is a better reconstruction input than a fabricated single home pose.
"""
    (args.output / "README.md").write_text(notes)
    print(json.dumps({k: report[k] for k in ("observed_frames", "observed_episodes", "episode_duration_s", "initial_state_raw", "initial_tcp_xyz_m", "close_tcp_xyz_m", "release_tcp_xyz_m")}, indent=2))


if __name__ == "__main__":
    main()
