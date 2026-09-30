"""Inspect and copy demonstrations ending at the first captured target light.

This module never edits source episodes or videos.  Its terminal action is the
last already-applied planned target, copied from source action[k-1], not the
measured terminal state or source action[k]'s unexecuted future target.
Only NumPy and the Python standard library are required.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from .panel_metadata import episode_panel_context, validate_episode_panel_metadata
import tempfile
import xml.etree.ElementTree as ET

import numpy as np


CUT_POLICY = "first_captured_target_light_inclusive_v1"
TERMINAL_ACTION_POLICY = "repeat_last_executed_planned_target_from_source_action_k_minus_1"
ROOT = Path(__file__).resolve().parents[2]
FRAME_SHAPES = {"state": (8,), "action": (8,), "sim_time": (), "q_actual": (6,),
                "q_target": (6,), "physics_index": (), "phase": (), "lights": (12,)}
PHYSICS_SHAPES = {"q_actual": (6,), "q_command": (6,), "gripper_actual": (2,),
                  "button_travel": (12,), "contact_force": (12,), "lights": (12,)}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def file_identity(path):
    """Compute a durable identity without loading an entire video into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": Path(path).stat().st_size, "sha256": digest.hexdigest()}


def _rotation(axis, angle):
    """Vectorized Rodrigues rotation; angle may contain several samples."""
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    angle = np.asarray(angle)[..., None, None]
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


@lru_cache(maxsize=8)
def _urdf_chain(path):
    root = ET.parse(path).getroot()
    joints = {j.find("child").get("link"): j for j in root.findall("joint")}
    chain, child = [], "link6"
    while child != "base_link":
        _require(child in joints and len(chain) < len(joints), "Invalid URDF base_link-to-link6 chain")
        joint = joints[child]
        origin = joint.find("origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
        rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
        _require(xyz.shape == rpy.shape == (3,) and np.isfinite(np.r_[xyz, rpy]).all(), "Invalid URDF origin")
        transform = np.eye(4)
        transform[:3, 3] = xyz
        transform[:3, :3] = (_rotation([0, 0, 1], rpy[2]) @ _rotation([0, 1, 0], rpy[1])
                             @ _rotation([1, 0, 0], rpy[0]))
        axis = None
        if joint.get("type") != "fixed":
            _require(joint.get("type") == "revolute", "Expected six revolute Piper arm joints")
            element = joint.find("axis")
            axis = np.fromstring(element.get("xyz", "1 0 0") if element is not None else "1 0 0", sep=" ")
            _require(axis.shape == (3,) and np.isfinite(axis).all() and np.linalg.norm(axis) > 0, "Invalid URDF axis")
            axis /= np.linalg.norm(axis)
        chain.append((transform, axis))
        child = joint.find("parent").get("link")
    chain.reverse()
    _require(sum(axis is not None for _, axis in chain) == 6, "Expected six Piper arm joints")
    return chain


def tcp_transforms(q, urdf_path):
    """Independent batched FK for the base-frame gripper TCP at local Z=.1358 m."""
    q = np.asarray(q, dtype=np.float64)
    _require(q.ndim == 2 and q.shape[1] == 6 and np.isfinite(q).all(), "Invalid joint array for TCP FK")
    transform = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
    index = 0
    for origin, axis in _urdf_chain(str(Path(urdf_path).resolve())):
        transform = transform @ origin
        if axis is not None:
            turn = np.broadcast_to(np.eye(4), transform.shape).copy()
            turn[:, :3, :3] = _rotation(axis, q[:, index])
            transform = transform @ turn
            index += 1
    transform[:, :3, 3] += transform[:, :3, 2] * .1358
    return transform


def _pose_errors(poses, joints, urdf_path):
    transforms = tcp_transforms(joints, urdf_path)
    quat = poses[:, 3:7]
    _require(np.allclose(np.linalg.norm(quat, axis=1), 1., rtol=0, atol=1e-7), "Non-unit pose quaternion")
    # Direct unit-quaternion rotation matrix; signs q and -q are equivalent.
    w, x, y, z = (quat / np.linalg.norm(quat, axis=1)[:, None]).T
    rotation = np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                         2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                         2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=1).reshape(-1, 3, 3)
    position_error = np.linalg.norm(poses[:, :3] - transforms[:, :3, 3], axis=1)
    rotation_error = np.max(np.abs(rotation - transforms[:, :3, :3]), axis=(1, 2))
    _require(position_error.max() < 1e-8 and rotation_error.max() < 1e-7,
             "Pose does not match base_link gripper_tcp FK of its recorded joints")
    return float(position_error.max()), float(rotation_error.max())


def inspect_press_prefix_arrays(frames, physics, metadata, collection, *, project_root=None):
    """Validate evidence and return a JSON-compatible cut record, without hashes.

    ``collection`` is collection_metadata.json (including its ``config``).
    This array-level entry point also supports small independent unit fixtures.
    """
    cfg = collection["config"]
    validate_episode_panel_metadata(collection, metadata)
    fps, hz, stride = (collection.get(k) for k in ("fps", "physics_hz", "capture_stride"))
    _require(type(fps) is int and fps > 0 and hz == 120 and 120 % fps == 0
             and type(stride) is int and stride == 120 // fps, "Invalid capture rates")
    _require(np.isclose(cfg["physics_dt"], 1 / hz, rtol=0, atol=1e-12), "Invalid physics dt")
    for source in (collection, metadata):
        _require(source.get("raw_schema_version") in (10, 11), "Press-prefix input requires raw schema 10 or 11")
        _require(source.get("raw_schema_version") == collection.get("raw_schema_version"), "Raw schema differs from collection")
        _require(all(source.get(k) == v for k, v in (("fps", fps), ("physics_hz", hz), ("capture_stride", stride)))
                 and np.isclose(source.get("action_horizon_s", -1), 1 / fps, rtol=0, atol=1e-12), "Inconsistent capture timing metadata")
        _require(source.get("pose_frame") == "base_link" and source.get("pose_link") == "gripper_tcp"
                 and source.get("tcp_offset_link6_m") == [0, 0, .1358], "Invalid TCP frame metadata")
    _require(metadata.get("collection_fingerprint") == collection["collection_fingerprint"], "Collection fingerprint mismatch")
    episode_id, floor = metadata["episode_id"], metadata["floor"]
    _require(type(episode_id) is int and episode_id >= 0 and type(floor) is int and 24 <= floor <= 35
             and floor == 24 + episode_id % 12, "Invalid source episode/floor identity")
    _require(metadata.get("success") is True and metadata.get("task") == f"Press {floor} floor."
             and not metadata.get("unexpected_collisions", [True]), "Source episode is not successful")
    n, p = metadata["num_frames"], metadata["physics_steps"]
    _require(type(n) is int and n >= 2 and type(p) is int and p == (n - 1) * stride + 1, "Invalid source lengths")
    for arrays, shapes, length in ((frames, FRAME_SHAPES, n), (physics, PHYSICS_SHAPES, p)):
        _require(set(arrays) == set(shapes), "Unexpected raw array schema")
        for key, shape in shapes.items():
            value = np.asarray(arrays[key])
            _require(value.shape == (length, *shape), f"Invalid {key} shape")
            _require(key == "phase" or np.isfinite(value).all(), f"Nonfinite {key}")
        _require(np.isin(arrays["lights"], [0, 1]).all(), "Lamp state is not binary")
    indices = np.arange(n) * stride
    _require(np.array_equal(frames["physics_index"], indices), "Frame/physics index mismatch")
    _require(np.allclose(frames["sim_time"], np.arange(n) / fps, rtol=0, atol=1e-10), "Frame timestamp mismatch")
    plan_steps = metadata["plan_physics_steps"]
    _require(type(plan_steps) is int and 1 <= plan_steps <= p and p - plan_steps < stride, "Invalid plan padding")
    future = np.minimum(indices + stride, plan_steps - 1)
    _require(np.array_equal(frames["q_target"], physics["q_command"][future]), "Source future q_target mismatch")
    _require(np.array_equal(frames["q_actual"], physics["q_actual"][indices]), "Measured joint/frame mismatch")
    _require(np.array_equal(frames["lights"], physics["lights"][indices]), "Lamp/frame mismatch")
    target = floor - 24
    lit = np.flatnonzero(frames["lights"][:, target])
    _require(len(lit) > 0, "No captured target light")
    k = int(lit[0])
    _require(k >= 1, "Target is already lit in the initial frame")
    end = int(indices[k])
    _require(end < plan_steps, "Cut is outside the original plan")
    prefix_lights = physics["lights"][:end + 1]
    _require(not np.delete(prefix_lights, target, axis=1).any(), "Wrong button lights before cut")
    on = np.flatnonzero(prefix_lights[:, target])
    _require(len(on) > 0, "Missing physical target press")
    first = int(on[0])
    _require((k - 1) * stride < first <= end, "First physical press is outside the first-lit sampling interval")
    _require(np.array_equal(prefix_lights[:, target], np.arange(end + 1) >= first), "Target released or flickered before cut")
    travel, force = physics["button_travel"][:end + 1], physics["contact_force"][:end + 1]
    trigger = (travel >= cfg["press_threshold"]) & (force > .02)
    _require(not np.delete(trigger, target, axis=1).any(), "Wrong button physically pressed before cut")
    triggered = np.flatnonzero(trigger[:, target])
    _require(len(triggered) > 0 and int(triggered[0]) == first, "Lamp onset disagrees with physical travel/force trigger")
    _require(np.all(travel[first:, target] > cfg["release_threshold"]), "Physical target release occurs before cut")
    events = [e for e in metadata["events"] if e["physics_index"] <= end]
    _require([(e["type"], e["floor"], e["physics_index"]) for e in events] == [("pressed", floor, first)], "Prefix events disagree with physics")
    _require(np.isclose(events[0]["time"], first / hz, rtol=0, atol=1e-10), "Press event timestamp mismatch")
    phases = frames["phase"][:k + 1].astype(str)
    _require(phases[-1] == "press", "First captured target light is not in the press phase")
    _require(not np.isin(phases, ["dwell", "retract", "return_home", "home_hold"]).any(), "Post-press phase appears before cut")
    _require(np.array_equal(frames["q_target"][k - 1], physics["q_command"][end]), "Terminal target is not the last executed command")
    urdf = Path(cfg["robot_urdf"])
    if not urdf.is_absolute():
        urdf = Path(project_root or ROOT) / urdf
    action_pos_error, action_rot_error = _pose_errors(frames["action"][:k], frames["q_target"][:k], urdf)
    state_pos_error, state_rot_error = _pose_errors(frames["state"][:k + 1], frames["q_actual"][:k + 1], urdf)
    width = float(collection.get("gripper_width_m", .008))
    _require(np.allclose(frames["action"][:k, 7], width, rtol=0, atol=1e-12), "Commanded gripper width mismatch")
    fingers = physics["gripper_actual"][indices[:k + 1]].astype(np.float64)
    _require(np.allclose(frames["state"][:k + 1, 7], fingers[:, 0] - fingers[:, 1], rtol=0, atol=1e-12), "Measured gripper width mismatch")
    return dict(source_episode_id=episode_id, floor=floor, task=metadata["task"],
                **episode_panel_context(metadata),
                cut_policy=CUT_POLICY, terminal_action_policy=TERMINAL_ACTION_POLICY,
                fps=fps, physics_hz=hz, capture_stride=stride, source_frames=n, source_physics_steps=p,
                cut_frame_index=k, cut_physics_index=end, kept_frames=k + 1, kept_physics_steps=end + 1,
                first_press_physics_index=first, first_press_time_s=first / hz,
                end_time_s=k / fps, end_sample_physics_time_s=(end + 1) / hz,
                capture_delay_from_first_press_s=(end - first) / hz, cut_phase=str(phases[-1]),
                removed_frames=n - k - 1, removed_physics_steps=p - end - 1,
                terminal_source_action_index=k - 1, terminal_source_q_target_index=k - 1,
                terminal_measured_state_used=False,
                terminal_action=frames["action"][k - 1].tolist(),
                terminal_q_target=frames["q_target"][k - 1].tolist(),
                terminal_action_state_position_difference_m=float(np.linalg.norm(frames["action"][k - 1, :3] - frames["state"][k, :3])),
                maximum_action_fk_position_error_m=action_pos_error,
                maximum_action_fk_rotation_matrix_error=action_rot_error,
                maximum_state_fk_position_error_m=state_pos_error,
                maximum_state_fk_rotation_matrix_error=state_rot_error)


def inspect_press_prefix(episode, collection, *, project_root=None):
    """Read one raw episode, verify all four source-file hashes, and inspect it."""
    directory = Path(episode)
    metadata = json.loads((directory / "metadata.json").read_text())
    _require(directory.name == f"episode_{metadata['episode_id']:06d}", "Episode directory identity mismatch")
    identities = {"metadata.json": file_identity(directory / "metadata.json")}
    for name in ("frames.npz", "physics.npz", "wrist.mp4", "global.mp4"):
        identities[name] = file_identity(directory / name)
        _require(identities[name] == metadata["source_files"].get(name), f"Source checksum mismatch: {name}")
    with np.load(directory / "frames.npz", allow_pickle=False) as source:
        frames = dict(source)
    with np.load(directory / "physics.npz", allow_pickle=False) as source:
        physics = dict(source)
    record = inspect_press_prefix_arrays(frames, physics, metadata, collection, project_root=project_root)
    record.update(source_episode_path=str(directory.resolve()), source_files=identities,
                  source_episode_sha256=hashlib.sha256(json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
    return record


def crop_press_prefix_arrays(frames, physics, cut_record):
    """Return independent copies of every retained row with terminal target clamp.

    Never shift the nonterminal actions: source action[i] already targets i+1.
    ``physics`` ends at the same post-physics sample as the final RGB frame.
    """
    k, end = cut_record["cut_frame_index"], cut_record["cut_physics_index"]
    _require(type(k) is int and k >= 1 and type(end) is int
             and end == k * cut_record["capture_stride"]
             and cut_record["kept_frames"] == k + 1 and cut_record["kept_physics_steps"] == end + 1,
             "Invalid cut record")
    _require(cut_record.get("cut_policy") == CUT_POLICY and cut_record.get("terminal_action_policy") == TERMINAL_ACTION_POLICY,
             "Unsupported cut policy")
    _require(set(frames) == set(FRAME_SHAPES) and set(physics) == set(PHYSICS_SHAPES), "Unexpected raw array schema")
    _require(all(len(a) == cut_record["source_frames"] for a in frames.values())
             and all(len(a) == cut_record["source_physics_steps"] for a in physics.values()), "Cut/source length mismatch")
    _require(np.array_equal(frames["action"][k - 1], cut_record["terminal_action"])
             and np.array_equal(frames["q_target"][k - 1], cut_record["terminal_q_target"])
             and np.array_equal(frames["q_target"][k - 1], physics["q_command"][end]), "Cut/source terminal target mismatch")
    _require(not frames["lights"][:k].any() and frames["lights"][k].sum() == 1
             and frames["lights"][k, cut_record["floor"] - 24] == 1, "Cut/source first-light mismatch")
    kept_frames = {key: np.asarray(value)[:k + 1].copy() for key, value in frames.items()}
    kept_physics = {key: np.asarray(value)[:end + 1].copy() for key, value in physics.items()}
    kept_frames["action"][-1] = frames["action"][k - 1]
    kept_frames["q_target"][-1] = frames["q_target"][k - 1]
    return kept_frames, kept_physics


def build_cut_plan(raw_root, *, project_root=None, episode_ids=None, episodes_per_floor=100):
    """Scan immutable source episodes; return an all-or-error derivation manifest."""
    raw_root = Path(raw_root).resolve()
    collection_path = raw_root / "collection_metadata.json"
    collection = json.loads(collection_path.read_text())
    ids = list(range(12 * episodes_per_floor)) if episode_ids is None else list(episode_ids)
    _require(len(ids) == len(set(ids)) == 12 * episodes_per_floor, "Invalid episode selection size")
    records = [inspect_press_prefix(raw_root / f"episode_{eid:06d}", collection, project_root=project_root) for eid in ids]
    counts = Counter(record["floor"] for record in records)
    _require(all(counts[floor] == episodes_per_floor for floor in range(24, 36)), "Floor count mismatch")
    collection_identity = file_identity(collection_path)
    return dict(schema_version=1, kind="press_prefix_cut_plan", success=True, created_at=datetime.now(timezone.utc).isoformat(),
                source_raw=str(raw_root), raw_root=str(raw_root), source_collection_fingerprint=collection["collection_fingerprint"],
                source_collection_metadata=collection_identity,
                source_collection_metadata_sha256=collection_identity["sha256"],
                source_raw_schema_version=collection["raw_schema_version"],
                inspection_code_sha256=file_identity(__file__)["sha256"],
                cut_policy=CUT_POLICY, terminal_action_policy=TERMINAL_ACTION_POLICY,
                fps=collection["fps"], physics_hz=collection["physics_hz"], capture_stride=collection["capture_stride"],
                action_horizon_s=collection["action_horizon_s"],
                time_origin="first captured post-physics state; sample physical time equals sim_time + 1/120",
                total_episodes=len(records), floor_counts={str(f): counts[f] for f in range(24, 36)},
                source_total_frames=sum(r["source_frames"] for r in records),
                kept_total_frames=sum(r["kept_frames"] for r in records),
                removed_total_frames=sum(r["removed_frames"] for r in records), episodes=records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-floor", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; reuse its exact bytes instead of changing the cut-plan hash")
    plan = build_cut_plan(args.raw, episodes_per_floor=args.episodes_per_floor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, suffix=".cut_plan.tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(plan, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, args.output)  # Atomic commit that fails if another plan already exists.
    finally:
        temporary.unlink()
    print(json.dumps({k: v for k, v in plan.items() if k != "episodes"}, indent=2))
