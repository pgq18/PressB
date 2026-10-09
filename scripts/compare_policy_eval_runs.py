#!/usr/bin/env python3
"""Compare complete closed-loop evaluation recordings using CPU only.

Example: python scripts/compare_policy_eval_runs.py --baseline-root OLD \
    --candidate-root NEW --output outputs/comparison

Outputs comparison.json and summary.md. A missing/failed worker or incompatible
evaluation identity produces an invalid comparison and exit code 2, never a
synthetic timeout. This is distinct from replaying identical inputs to inference.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


REASONS = {"target_pressed": "目标按亮", "wrong_button_pressed": "误按",
           "time_limit": "超时", "unexpected_collision": "异常碰撞",
           "invalid_policy_action": "无效策略动作"}
RUNTIME_KEYS = ("runtime", "environment", "software_versions", "engine_version",
                "isaac_sim_version", "isaacsim_version", "runtime_versions",
                "renderer_settings", "robot_urdf")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence(path, inventory):
    path = Path(path).resolve()
    inventory[str(path)] = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def read(path, inventory):
    evidence(path, inventory)
    return json.loads(Path(path).read_text())


def finite(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f"{label}: expected finite number")
    return float(value)


def condition(row):
    for key in ("floor", "repeat"):
        require(type(row.get(key)) is int, f"Condition needs integer {key}")
    require(isinstance(row.get("panel_layout_name"), str), "Missing panel layout name")
    return (row["floor"], row["repeat"], row["panel_layout_name"],
            finite(row["panel_offset_x_m"], "panel x"),
            finite(row["panel_offset_y_m"], "panel y"))


def condition_record(key):
    return dict(zip(("floor", "repeat", "panel_layout_name", "panel_offset_x_m",
                     "panel_offset_y_m"), key))


def unique_ids(rows, label):
    result = {}
    for row in rows:
        index = row["episode_id"]
        require(type(index) is int and index >= 0, f"{label}: invalid episode id")
        require(index not in result, f"{label}: duplicate episode id {index}")
        result[index] = row
    return result


def first_request(path, inventory):
    evidence(path, inventory)
    with path.open() as stream:
        line = next((line for line in stream if line.strip()), None)
    require(line is not None, f"{path}: empty requests")
    request = json.loads(line)
    require(request["chunk_index"] == request["observation_physics_index"] == 0
            and request["observation_sim_time"] == 0, f"{path}: first observation is not initial")
    return request


def identity(manifest, service, report):
    args = manifest["arguments"]
    checkpoint = manifest["expected_checkpoint_sha256"]
    require(checkpoint == args["expected_checkpoint_sha256"] == service["model_sha256"]
            == service["provenance"]["model_sha256"], "Checkpoint SHA mismatch within worker")
    step = manifest["expected_checkpoint_step"]
    require(step == args["expected_checkpoint_step"] == service["checkpoint_step"],
            "Checkpoint step mismatch within worker")
    require(service["status"] == "ready" and service["checkpoint_verified"] is True,
            "Policy service checkpoint was not verified")
    require(manifest["motion_smoothing"] == report["motion_smoothing"]
            and manifest["motion_smoothing"]["window"] == args["smoothing_window"],
            "Motion smoothing mismatch within worker")
    require(args["max_seconds"] == report["max_sim_seconds"], "Duration mismatch within worker")
    require(report["fps"] == service["fps"] and report["action_chunk_size"] == service["action_horizon"],
            "Policy frequency/horizon mismatch within worker")
    return dict(checkpoint_sha256=checkpoint, checkpoint_step=step,
                scene_sha256=manifest["scene_sha256"], config=manifest["config"],
                base_seed=args["seed"], motion_smoothing=manifest["motion_smoothing"],
                max_sim_seconds=args["max_seconds"], fps=report["fps"],
                action_chunk_size=report["action_chunk_size"],
                panel_layout_mode=manifest["panel_layout_mode"],
                panel_layouts=manifest["panel_layouts"],
                inference_seed_rule=manifest["inference_seed_rule"],
                inference_clock=manifest["inference_clock"],
                policy_contract={key: service[key] for key in (
                    "camera_order", "resolution", "input_image_resize", "num_inference_timesteps",
                    "pose_frame", "pose_link", "xyz_units", "rotation6d", "quaternion_order",
                    "gripper_is_learned", "controller_gripper_width_m")})


def read_run(root):
    root = Path(root).resolve()
    workers = ([root] if (root / "eval_manifest.json").is_file() else
               sorted(path for path in root.glob("worker_*") if path.is_dir()))
    require(workers, f"{root}: no evaluation workers found")
    common, episodes, inventory, deployments = None, {}, {}, []
    for worker in workers:
        docs = {name: read(worker / filename, inventory) for name, filename in {
            "manifest": "eval_manifest.json", "service": "policy_service.json",
            "report": "report.json", "status": "status.json"}.items()}
        manifest, service, report, status = (docs[key] for key in ("manifest", "service", "report", "status"))
        require(status.get("status") == "complete" and report.get("complete") is True,
                f"{worker}: worker execution is incomplete or failed")
        contract = identity(manifest, service, report)
        require(common is None or contract == common, f"{worker}: workers disagree on evaluation identity")
        common = contract
        schedules = unique_ids(manifest["episode_schedule"], str(worker / "eval_manifest.json"))
        reports = unique_ids(report["episodes"], str(worker / "report.json"))
        require(schedules and schedules.keys() == reports.keys(), f"{worker}: missing or extra report episodes")
        expected_dirs = {f"episode_{index:06d}" for index in schedules}
        actual_dirs = {path.name for path in worker.glob("episode_*") if path.is_dir()}
        require(expected_dirs == actual_dirs, f"{worker}: missing or extra episode directories")
        require(report["total_episodes"] == status["completed_episodes"] == len(schedules),
                f"{worker}: completion counts do not match schedule")
        audit_summary = None
        if (worker / "audit.json").is_file():
            audit = read(worker / "audit.json", inventory)
            require(audit.get("audit_pass") is True and not audit.get("errors")
                    and audit.get("audited_episodes") == len(schedules), f"{worker}: independent audit failed/incomplete")
            audit_summary = {key: audit[key] for key in ("audit_pass", "audited_episodes", "errors")}
        deployment = dict(worker=worker.name, root=str(worker), arguments=manifest["arguments"],
                          checkpoint_path=service["provenance"]["checkpoint"],
                          execution_sources=manifest.get("sources", {}), independent_audit=audit_summary,
                          recorded_runtime={key: manifest[key] for key in RUNTIME_KEYS if key in manifest})
        for filename in ("runtime.json", "environment.json", "software_versions.json"):
            if (worker / filename).is_file():
                deployment["recorded_runtime"][filename] = read(worker / filename, inventory)
        deployments.append(deployment)
        for index, schedule in schedules.items():
            episode_path = worker / f"episode_{index:06d}"
            metadata = read(episode_path / "metadata.json", inventory)
            row = reports[index]
            key = condition(schedule)
            require(key == condition(row) == condition(metadata), f"{episode_path}: condition mismatch")
            require(key not in episodes, f"{root}: duplicate condition {key}")
            require(metadata["episode_id"] == index, f"{episode_path}: incorrect metadata episode id")
            for field in ("success", "task_success", "termination", "sim_seconds", "min_target_tip_distance_m"):
                require(row[field] == metadata[field], f"{episode_path}: report/metadata disagree on {field}")
            for field in ("success", "task_success"):
                require(type(metadata[field]) is bool, f"{episode_path}: invalid {field}")
            require(metadata["termination"] in REASONS, f"{episode_path}: unknown termination")
            require(not metadata.get("error") or metadata["termination"] == "invalid_policy_action",
                    f"{episode_path}: episode execution error cannot be counted as a normal outcome")
            require(metadata["success"] == (metadata["termination"] == "target_pressed"),
                    f"{episode_path}: inconsistent success/termination")
            require(metadata["config"] == common["config"]
                    and metadata["motion_smoothing"] == common["motion_smoothing"],
                    f"{episode_path}: episode configuration differs from manifest")
            require(metadata["seed_episode_index"] == schedule["seed_episode_index"],
                    f"{episode_path}: episode seed differs from schedule")
            seconds = finite(metadata["sim_seconds"], "episode duration")
            distance = finite(metadata["min_target_tip_distance_m"], "target distance")
            require(0 <= seconds <= common["max_sim_seconds"] and distance >= 0,
                    f"{episode_path}: invalid duration/distance")
            require(metadata["termination"] != "time_limit" or
                    math.isclose(seconds, common["max_sim_seconds"], abs_tol=1e-7),
                    f"{episode_path}: reported timeout before time limit")
            require(isinstance(metadata["unexpected_collisions"], list), f"{episode_path}: invalid collision evidence")
            request = first_request(episode_path / "requests.jsonl", inventory)
            require(request["seed"] == common["base_seed"] + schedule["seed_episode_index"] * 10000,
                    f"{episode_path}: first request seed differs from schedule")
            require(request["task"] == row["task"] == metadata["task"], f"{episode_path}: task mismatch")
            pressed_floors = [event["floor"] for event in metadata["events"] if event["type"] == "pressed"]
            require(metadata["task_success"] == (pressed_floors == [metadata["floor"]]),
                    f"{episode_path}: task success differs from physical button events")
            require(not metadata["success"] or (metadata["task_success"] and not metadata["unexpected_collisions"]),
                    f"{episode_path}: successful outcome lacks a clean target press")
            episodes[key] = dict(worker=worker.name, episode_id=index, path=episode_path, metadata=metadata,
                                 request=request, seed_episode_index=schedule["seed_episode_index"],
                                 pressed_floors=pressed_floors)
        require(sum(reports[index]["success"] for index in reports) == report["passed_episodes"],
                f"{worker}: success count does not match episodes")
    return dict(root=str(root), identity=common, episodes=episodes, deployments=deployments, input_files=inventory)


def numeric_array(value, shape, label):
    result = np.asarray(value, dtype=np.float64)
    require(result.shape == shape and np.isfinite(result).all(), f"{label}: invalid shape/nonfinite values")
    return result


def request_metrics(baseline, candidate, horizon):
    left, right = baseline["request"], candidate["request"]
    require(left["seed"] == right["seed"] and left["task"] == right["task"], "Initial request seed/task mismatch")
    a = numeric_array(left["state"], (8,), "baseline state")
    b = numeric_array(right["state"], (8,), "candidate state")
    qa, qb = a[3:7], b[3:7]
    require(np.linalg.norm(qa) > 1e-12 and np.linalg.norm(qb) > 1e-12, "Zero state quaternion")
    dot = float(np.dot(qa / np.linalg.norm(qa), qb / np.linalg.norm(qb)))
    rotation = 2 * math.acos(min(1., abs(dot)))
    actions_a = numeric_array(left["response"]["actions_pose9"], (horizon, 9), "baseline prediction")
    actions_b = numeric_array(right["response"]["actions_pose9"], (horizon, 9), "candidate prediction")
    diff = actions_b[:, :3] - actions_a[:, :3]
    return dict(seed=left["seed"], state_position_error_m=float(np.linalg.norm(b[:3] - a[:3])),
                state_rotation_error_rad=rotation, state_rotation_error_deg=math.degrees(rotation),
                state_gripper_width_delta_m=float(b[7] - a[7]),
                first_prediction_xyz_rmse_m=float(np.sqrt(np.mean(diff ** 2))),
                first_prediction_xyz_max_abs_error_m=float(np.max(np.abs(diff))))


def initial_image(episode, camera, inventory):
    record = episode["request"]["images"][camera]
    relative = Path(record["path"])
    require(not relative.is_absolute(), "Initial observation path must be episode-relative")
    path = (episode["path"] / relative).resolve()
    require(path.is_relative_to(episode["path"].resolve()), "Initial observation escapes episode directory")
    if not path.is_file():
        return None, dict(available=False, path=str(path), reason="lossless_observation_not_retrieved")
    evidence(path, inventory)
    if record.get("sha256"):
        require(inventory[str(path)]["sha256"] == record["sha256"], f"{path}: initial image file SHA mismatch")
    with Image.open(path) as source:
        require(source.format == "PNG", f"{path}: require lossless PNG observation, not video/JPEG preview")
        image = source.convert("RGB")
    pixels = np.asarray(image)
    if record.get("rgb_sha256"):
        require(hashlib.sha256(pixels.tobytes()).hexdigest() == record["rgb_sha256"], f"{path}: RGB input SHA mismatch")
    return image, dict(available=True, path=str(path), size=list(image.size), sha256=inventory[str(path)]["sha256"])


def image_metrics(baseline, candidate, camera, baseline_inventory, candidate_inventory, resolution):
    a, a_record = initial_image(baseline, camera, baseline_inventory)
    b, b_record = initial_image(candidate, camera, candidate_inventory)
    result = dict(baseline=a_record, candidate=b_record, comparable=a is not None and b is not None)
    if not result["comparable"]:
        return result
    require(a.size == b.size, f"{camera}: native observation dimensions differ")
    delta = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    a_small = np.asarray(a.resize((resolution, resolution), Image.Resampling.BILINEAR), dtype=np.float64)
    b_small = np.asarray(b.resize((resolution, resolution), Image.Resampling.BILINEAR), dtype=np.float64)
    result.update(rgb_mae=float(np.mean(np.abs(delta))), rgb_rmse=float(np.sqrt(np.mean(delta ** 2))),
                  resized_rgb_mae=float(np.mean(np.abs(b_small - a_small))),
                  resized_rgb_rmse=float(np.sqrt(np.mean((b_small - a_small) ** 2))),
                  resize=[resolution, resolution], pixel_scale="0..255 RGB", resize_filter="PIL bilinear")
    return result


def summary_stats(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return dict(count=0, mean=None, mean_abs=None, min=None, max=None, max_abs=None)
    return dict(count=len(values), mean=float(np.mean(values)), mean_abs=float(np.mean(np.abs(values))),
                min=float(np.min(values)), max=float(np.max(values)), max_abs=float(np.max(np.abs(values))))


def outcome_summary(run):
    metadata = [entry["metadata"] for entry in run["episodes"].values()]
    counts = Counter(item["termination"] for item in metadata)
    return dict(episodes=len(metadata), successes=sum(item["success"] for item in metadata),
                success_rate=sum(item["success"] for item in metadata) / len(metadata),
                wrong_button_presses=counts["wrong_button_pressed"], timeouts=counts["time_limit"],
                collision_episodes=sum(bool(item["unexpected_collisions"]) for item in metadata),
                collision_events=sum(len(item["unexpected_collisions"]) for item in metadata),
                invalid_policy_actions=counts["invalid_policy_action"], termination_counts=dict(counts))


def deployment_differences(baseline, candidate, rows):
    """List allowed differences for workers that contributed matching conditions."""
    left = {row["worker"]: row for row in baseline["deployments"]}
    right = {row["worker"]: row for row in candidate["deployments"]}
    pairs = sorted({(row["baseline"]["worker"], row["candidate"]["worker"]) for row in rows})
    result = []
    for before, after in pairs:
        differences = {}
        for key in ("root", "arguments", "checkpoint_path", "execution_sources", "recorded_runtime"):
            a, b = left[before][key], right[after][key]
            if a != b:
                differences[key] = dict(baseline=a, candidate=b)
        result.append(dict(baseline_worker=before, candidate_worker=after, differences=differences))
    return result


def compare(baseline_root, candidate_root):
    a, b = read_run(baseline_root), read_run(candidate_root)
    mismatches = [key for key in a["identity"] if a["identity"][key] != b["identity"][key]]
    require(not mismatches, f"Evaluation identity mismatch: {', '.join(mismatches)}")
    require(a["episodes"].keys() == b["episodes"].keys(),
            f"Condition coverage mismatch; missing={sorted(a['episodes'].keys() - b['episodes'].keys())}; "
            f"extra={sorted(b['episodes'].keys() - a['episodes'].keys())}")
    rows = []
    for key in sorted(a["episodes"]):
        left, right = a["episodes"][key], b["episodes"][key]
        lm, rm = left["metadata"], right["metadata"]
        require(left["seed_episode_index"] == right["seed_episode_index"], f"{key}: per-condition seed mismatch")
        row = dict(condition_record(key), baseline={field: lm[field] for field in (
            "success", "termination", "sim_seconds", "min_target_tip_distance_m")},
            candidate={field: rm[field] for field in (
            "success", "termination", "sim_seconds", "min_target_tip_distance_m")})
        for label, entry in (("baseline", left), ("candidate", right)):
            row[label].update(worker=entry["worker"], episode_id=entry["episode_id"],
                              pressed_floors=entry["pressed_floors"],
                              collision_events=len(entry["metadata"]["unexpected_collisions"]))
        row.update(termination_matches=lm["termination"] == rm["termination"],
                   pressed_floors_match=left["pressed_floors"] == right["pressed_floors"],
                   success_changed=lm["success"] != rm["success"],
                   termination_time_delta_s=rm["sim_seconds"] - lm["sim_seconds"],
                   min_target_tip_distance_delta_m=rm["min_target_tip_distance_m"] - lm["min_target_tip_distance_m"],
                   initial_request=request_metrics(left, right, a["identity"]["action_chunk_size"]),
                   initial_images={camera: image_metrics(left, right, camera, a["input_files"], b["input_files"],
                       a["identity"]["policy_contract"]["resolution"]) for camera in ("global", "wrist")})
        row["termination_and_pressed_floors_match"] = row["termination_matches"] and row["pressed_floors_match"]
        rows.append(row)
    numeric = {key: summary_stats([row[key] for row in rows]) for key in (
        "termination_time_delta_s", "min_target_tip_distance_delta_m")}
    numeric.update({key: summary_stats([row["initial_request"][key] for row in rows]) for key in (
        "state_position_error_m", "state_rotation_error_deg", "first_prediction_xyz_rmse_m")})
    image_summary = {camera: {metric: summary_stats([row["initial_images"][camera][metric] for row in rows
                    if row["initial_images"][camera]["comparable"]]) for metric in (
                    "rgb_mae", "rgb_rmse", "resized_rgb_mae", "resized_rgb_rmse")} for camera in ("global", "wrist")}
    return dict(schema_version=1, comparison_valid=True, created_at=datetime.now(timezone.utc).isoformat(),
                baseline_root=a["root"], candidate_root=b["root"], identity=a["identity"],
                baseline=outcome_summary(a), candidate=outcome_summary(b),
                agreement={key: sum(row[key] for row in rows) for key in (
                    "termination_matches", "pressed_floors_match", "termination_and_pressed_floors_match")},
                success_changes=[row for row in rows if row["success_changed"]],
                metrics=numeric, image_metrics=image_summary, episodes=rows,
                deployment_records=dict(baseline=a["deployments"], candidate=b["deployments"]),
                deployment_differences=deployment_differences(a, b, rows),
                input_files=dict(baseline=list(a["input_files"].values()), candidate=list(b["input_files"].values())),
                limitations=["Closed-loop comparison, not identical-input inference replay; initial images/state can differ.",
                             "Missing lossless observations are listed explicitly and omitted only from image metrics.",
                             "Machine paths, GPU assignment and execution-source hashes may differ; see deployment_records.",
                             "Unrecorded engine versions cannot be inferred from the GPU or execution-source hashes.",
                             "Difference signs are candidate minus baseline. Pixel metrics use RGB intensities 0..255.",
                             "This compares recorded outcomes; it does not replace independent physical/video audits."])


def markdown(result):
    if not result["comparison_valid"]:
        return "# Eval 对比未通过完整性校验\n\n" + result["error"] + "\n\n未生成成功率或超时统计；缺失/失败运行未计为超时。\n"
    a, b = result["baseline"], result["candidate"]
    lines = ["# 跨机器闭环 Eval 对比", "", f"已按条件匹配 {a['episodes']} 条 episode；checkpoint、场景 SHA、配置、seed、平滑及运行时限一致。", "",
             "| 结果 | 本机基线 | 候选运行 |", "|---|---:|---:|"]
    for name, key in (("成功", "successes"), ("误按", "wrong_button_presses"), ("超时", "timeouts"),
                      ("发生异常碰撞的 episode", "collision_episodes"), ("无效策略动作", "invalid_policy_actions")):
        lines.append(f"| {name} | {a[key]} | {b[key]} |")
    agreement = result["agreement"]["termination_and_pressed_floors_match"]
    lines.extend(["", f"终止原因与实际按下楼层同时一致：**{agreement}/{a['episodes']}**。", "",
                  "| 差异指标 | 平均绝对差 | 最大绝对差 |", "|---|---:|---:|"])
    for key, label, scale in (("termination_time_delta_s", "终止时间（秒）", 1),
                              ("min_target_tip_distance_delta_m", "最近目标距离（mm）", 1000),
                              ("state_position_error_m", "初始 TCP 位置（mm）", 1000),
                              ("state_rotation_error_deg", "初始 TCP 姿态（度）", 1),
                              ("first_prediction_xyz_rmse_m", "首个预测 chunk 的 XYZ RMSE（mm）", 1000)):
        data = result["metrics"][key]
        lines.append(f"| {label} | {data['mean_abs'] * scale:.6g} | {data['max_abs'] * scale:.6g} |")
    lines.extend(["", "图像使用无损初始输入，数值单位为 0–255 像素强度：", "",
                  "| 相机 | 可比较条数 | 原图平均 MAE | 原图平均 RMSE | 224×224 平均 MAE |", "|---|---:|---:|---:|---:|"])
    for camera, data in result["image_metrics"].items():
        values = ["未取回" if data[key]["mean"] is None else f"{data[key]['mean']:.6g}"
                  for key in ("rgb_mae", "rgb_rmse", "resized_rgb_mae")]
        lines.append(f"| {camera} | {data['rgb_mae']['count']} | " + " | ".join(values) + " |")
    lines.extend(["", "成功状态变化："])
    if result["success_changes"]:
        lines.append("")
        for row in result["success_changes"]:
            lines.append(f"- {row['floor']} 楼 / {row['panel_layout_name']} / repeat {row['repeat']}："
                         f"{REASONS[row['baseline']['termination']]} → {REASONS[row['candidate']['termination']]}")
    else:
        lines.extend(["", "无。"])
    lines.extend(["", "本结果是闭环行为对比，首帧状态和渲染差异也会影响预测；不包含相同输入的远程推理复测。",
                  "机器路径、GPU、源码哈希及已记录的软件版本差异见 comparison.json 的 deployment_differences；未记录的软件版本不作推断。",
                  "全部逐条件差异及输入文件 SHA256 见 comparison.json。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Directory for comparison.json and summary.md")
    args = parser.parse_args()
    try:
        result = compare(args.baseline_root, args.candidate_root)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        result = dict(schema_version=1, comparison_valid=False, error=f"{type(exc).__name__}: {exc}",
                      baseline_root=str(args.baseline_root.resolve()), candidate_root=str(args.candidate_root.resolve()))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (args.output / "summary.md").write_text(markdown(result))
    print(json.dumps({key: result[key] for key in ("comparison_valid", "baseline", "candidate", "agreement", "error")
                      if key in result}, ensure_ascii=False))
    return 0 if result["comparison_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
