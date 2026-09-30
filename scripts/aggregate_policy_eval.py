#!/usr/bin/env python3
"""Combine four independently audited 15-condition policy workers, using CPU only.

Audit validity and physical task success are separate outcomes. This script
does not rerun inference, alter recordings, or classify RGB image semantics.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
from pathlib import Path


PANEL_FIELDS = ("panel_layout_index", "panel_layout_name", "panel_offset_x_m", "panel_offset_y_m")
LAYOUTS = [("center", 0., 0.), ("xmin_ymin", -.01, -.025), ("xmin_ymax", -.01, .025),
           ("xmax_ymin", .01, -.025), ("xmax_ymax", .01, .025)]
REASONS = {"target_pressed": "目标按亮", "time_limit": "超时", "wrong_button_pressed": "误按",
           "unexpected_collision": "异常碰撞", "invalid_policy_action": "无效动作"}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def index(rows):
    result = {r["episode_id"]: r for r in rows}
    require(len(result) == len(rows), "Duplicate local episode identity")
    return result


def statistics(rows):
    return {"episodes": len(rows), "successes": sum(r["success"] for r in rows),
            "task_successes": sum(r["task_success"] for r in rows),
            "success_rate": sum(r["success"] for r in rows) / len(rows) if rows else None,
            "termination_counts": dict(Counter(r["termination"] for r in rows))}


def aggregate(root, expected_sha, expected_step=10600):
    root = Path(root).resolve()
    require(len(expected_sha) == 64 and all(c in "0123456789abcdef" for c in expected_sha), "Invalid expected model SHA256")
    expected_workers = [f"worker_{i}" for i in range(4)]
    require(sorted(p.name for p in root.glob("worker_*") if p.is_dir()) == expected_workers, "Expected exactly worker_0..3")
    common, rows, workers, hashes_checked = None, [], [], 0
    expected_layouts = [dict(panel_layout_index=i, panel_layout_name=name, panel_offset_x_m=x, panel_offset_y_m=y)
                        for i, (name, x, y) in enumerate(LAYOUTS)]
    for worker_name in expected_workers:
        worker = root / worker_name
        paths = {name: worker / filename for name, filename in {
            "manifest": "eval_manifest.json", "service": "policy_service.json", "report": "report.json",
            "audit": "audit.json", "diagnosis": "diagnosis.json", "status": "status.json"}.items()}
        records = {name: read(path) for name, path in paths.items()}
        manifest, service, report, audit, diagnosis, status = (records[k] for k in paths)
        args, prov = manifest["arguments"], audit["provenance"]
        require(status.get("status") == "complete" and report.get("complete") is True, f"{worker_name} execution incomplete")
        require(audit.get("audit_pass") is True and not audit.get("errors"), f"{worker_name} independent audit failed")
        require(diagnosis.get("run_complete") is True, f"{worker_name} diagnosis is incomplete")
        require(audit["audited_episodes"] == diagnosis["diagnosed_episodes"] == report["total_episodes"] == 15,
                f"{worker_name} must contain fifteen fully audited conditions")
        require(manifest["expected_checkpoint_sha256"] == args["expected_checkpoint_sha256"] ==
                service["model_sha256"] == service["provenance"]["model_sha256"] ==
                prov["checkpoint_sha256"] == diagnosis["checkpoint_sha256"] == expected_sha, "Checkpoint SHA256 mismatch")
        require(manifest["expected_checkpoint_step"] == args["expected_checkpoint_step"] == service["checkpoint_step"] ==
                prov["checkpoint_step"] == diagnosis["checkpoint_step"] == expected_step, "Checkpoint step mismatch")
        require(service["status"] == "ready" and service["checkpoint_verified"] is True
                and service["fps"] == report["fps"] == 30 and service["action_horizon"] == 7
                and service["camera_order"] == ["global", "wrist"], "Policy service/execution contract mismatch")
        for name, field in (("manifest", "eval_manifest_sha256"), ("service", "policy_service_sha256")):
            require(prov[field] == diagnosis[field] == sha(paths[name]), f"Audited {name} changed")
        require(Path(audit["run"]).resolve() == Path(diagnosis["run"]).resolve() == worker, "Worker report refers to another run")
        require(manifest["panel_layout_mode"] == report["panel_layout_mode"] == prov["panel_layout_mode"] == "center_corners"
                and manifest["panel_layouts"] == report["panel_layouts"] == prov["panel_layouts"] == expected_layouts,
                "Panel layout table differs from the requested five positions")
        require(args["panel_layouts"] == "center_corners" and args["episodes_per_floor"] == 1
                and args["max_seconds"] == report["max_sim_seconds"] == 15., "Wrong repeats or duration")
        require(args["smoothing_window"] == manifest["motion_smoothing"]["window"] == 3,
                "Expected window-three smoothing")
        require(manifest["motion_smoothing"] == report["motion_smoothing"] == audit["motion_smoothing"] == diagnosis["motion_smoothing"],
                "Smoothing differs within a worker")
        require(manifest["recorded_actions_used"] is False and manifest["target_planner_used"] is False,
                "Run declares expert action/target planning")
        collection_path = Path(args["dataset"]) / "meta/collection_metadata.json"
        collection = read(collection_path)
        fingerprint = hashlib.sha256(json.dumps(collection["identity"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        require(prov["dataset_collection_sha256"] == sha(collection_path), "Training collection changed after audit")
        require(manifest["collection_fingerprint"] == prov["collection_fingerprint"] == collection["collection_fingerprint"] == fingerprint,
                "Collection fingerprint mismatch")
        require(manifest["config"] == collection["config"] == collection["identity"]["config"], "Configuration differs from training")
        require(manifest["scene_sha256"] == prov["scene_sha256"] == collection["scene_sha256"] == sha(args["snapshot"]),
                "Frozen training scene mismatch")
        signature = dict(checkpoint_sha256=expected_sha, checkpoint_step=expected_step,
                         config=manifest["config"], scene_sha256=manifest["scene_sha256"], collection_fingerprint=fingerprint,
                         collection_metadata_sha256=sha(collection_path), motion_smoothing=manifest["motion_smoothing"],
                         fps=report["fps"], action_chunk_size=report["action_chunk_size"],
                         base_seed=args["seed"], max_sim_seconds=args["max_seconds"], panel_layouts=expected_layouts,
                         execution_sources=manifest["sources"], audit_source_sha256=audit["audit_source_sha256"],
                         diagnostic_source_sha256=diagnosis["diagnostic_source_sha256"])
        if common is None:
            common = signature
        else:
            require(signature == common, "Worker model/configuration/scene/smoothing/fps/seed/source identity differs")
        floors = args["floors"]
        require(len(floors) == len(set(floors)) == 3 and all(type(f) is int and 24 <= f <= 35 for f in floors),
                "Each worker must select three distinct supported floors")
        schedule = [dict(episode_id=i * 3 + j, floor=f, repeat=0, seed_episode_index=i * 12 + f - 24, **layout)
                    for i, layout in enumerate(expected_layouts) for j, f in enumerate(floors)]
        require(manifest["episode_schedule"] == schedule, "Declared schedule differs from independently rebuilt schedule")
        reported, audited, diagnosed = (index(records[name]["episodes"]) for name in ("report", "audit", "diagnosis"))
        require(set(reported) == set(audited) == set(diagnosed) == set(range(15)), "Missing or extra worker episode IDs")
        require({p.name for p in worker.glob("episode_*") if p.is_dir()} == {f"episode_{i:06d}" for i in range(15)},
                "Unreferenced or missing episode directories")
        for planned in schedule:
            eid = planned["episode_id"]
            r, a, d = reported[eid], audited[eid], diagnosed[eid]
            directory = worker / f"episode_{eid:06d}"
            for record in (r, a, d):
                require(all(record[key] == planned[key] for key in ("floor", "repeat", *PANEL_FIELDS)), "Episode condition mismatch")
                require(type(record["success"]) is bool and record["success"] == a["success"]
                        and record["task_success"] == a["task_success"]
                        and record["termination"] == a["termination"], "Task outcome differs between reports")
            require(a["audit_pass"] is True and type(a["task_success"]) is bool and r["task_success"] == a["task_success"],
                    "Invalid audited episode outcome")
            require(a["seed_episode_index"] == planned["seed_episode_index"] and a["checked_inference_base_seed"] == args["seed"],
                    "Audited inference seed differs from its condition")
            require(set(a["files_sha256"]) == {"metadata.json", "physics.npz", "frames.npz", "requests.jsonl", "actions.jsonl"}
                    and set(d["sources"]) == {"metadata.json", "physics.npz", "requests.jsonl", "actions.jsonl"},
                    "Incomplete audited/diagnosed source inventory")
            for name, digest in a["files_sha256"].items():
                require(sha(directory / name) == digest, f"Audited episode file changed: {directory/name}")
                hashes_checked += 1
            for view in ("global", "wrist"):
                require(sha(directory / f"{view}.mp4") == a["videos"][view]["sha256"], "Audited video changed")
                hashes_checked += 1
            for name, digest in d["sources"].items():
                require(a["files_sha256"][name] == digest, "Diagnosis used a different episode source")
            metadata = read(directory / "metadata.json")
            require(all(metadata[k] == v for k, v in planned.items()), "Episode metadata differs from condition schedule")
            require(metadata["config"] == common["config"] and metadata["motion_smoothing"] == common["motion_smoothing"],
                    "Episode execution configuration differs")
            requests = [json.loads(line) for line in (directory / "requests.jsonl").read_text().splitlines() if line.strip()]
            require(len(requests) == a["requests"], "Request count changed")
            for request in requests:
                for image in request["images"].values():
                    path = directory / image["path"]
                    require(path.resolve().is_relative_to(directory), "Policy image path escapes the episode")
                    require(sha(path) == image["sha256"], "Audited policy input PNG changed")
                    hashes_checked += 1
            rows.append(dict(worker=worker_name, local_episode_id=eid, condition_index=planned["seed_episode_index"],
                             floor=planned["floor"], repeat=0, **{k: planned[k] for k in PANEL_FIELDS},
                             success=a["success"], task_success=a["task_success"], audit_pass=a["audit_pass"],
                             termination=a["termination"], sim_seconds=a["simulation_seconds"],
                             min_target_tip_distance_mm=a["min_target_tip_distance_m"] * 1000,
                             final_target_tip_distance_mm=a["final_target_tip_distance_m"] * 1000,
                             pressed_floors=[event["floor"] for event in a["events"] if event["type"] == "pressed"],
                             unexpected_collisions=a["unexpected_collisions"],
                             max_command_projection_mm=a["max_command_projection_m"] * 1000,
                             global_video=str((directory / "global.mp4").relative_to(root)),
                             wrist_video=str((directory / "wrist.mp4").relative_to(root))))
        local_rows = rows[-15:]
        require(sum(r["success"] for r in local_rows) == report["passed_episodes"] == audit["collision_free_successes"]
                and sum(r["task_success"] for r in local_rows) == report["task_successes"] == audit["task_successes"],
                "Reported aggregate outcomes disagree with episode evidence")
        workers.append(dict(worker=worker_name, audit_pass=audit["audit_pass"], **statistics(local_rows),
                            sources={name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()}))
    conditions = [(r["floor"], r["panel_layout_index"], r["repeat"]) for r in rows]
    require(len(conditions) == len(set(conditions)) == 60 and set(conditions) == {
        (floor, layout, 0) for floor in range(24, 36) for layout in range(5)}, "Missing or duplicate floor/layout condition across workers")
    rows.sort(key=lambda row: row["condition_index"])
    require([row["condition_index"] for row in rows] == list(range(60)), "Combined inference conditions are incomplete")
    return dict(schema_version=1, complete=True, aggregation_pass=True, all_worker_audits_pass=True,
                created_at=datetime.now(timezone.utc).isoformat(), root=str(root),
                aggregate_source_sha256=sha(Path(__file__)), identity=common, workers=workers,
                overall=statistics(rows), by_floor={str(f): statistics([r for r in rows if r["floor"] == f]) for f in range(24, 36)},
                by_position={name: statistics([r for r in rows if r["panel_layout_name"] == name]) for name, _, _ in LAYOUTS},
                episodes=rows, rehashed_audited_artifact_count=hashes_checked,
                limitations=["Passing evidence audits does not imply successful button presses; success counts come from audited physical contact outcomes.",
                             "RGB input hashes, frame counts and timestamps are audited; image semantics and visible orange-light meaning are not automatically classified.",
                             "Panel camera validation establishes frustum containment, not absence of robot occlusion.",
                             "Each floor/position has one rollout; this is a fixed 60-condition evaluation, not a statistical estimate over arbitrary layouts.",
                             "Simulation pauses for inference; these outcomes do not measure real-time deployment latency."])


def summary_markdown(report):
    overall, identity = report["overall"], report["identity"]
    lines = ["# VLA-JEPA 面板位置变化评估", "",
             f"Step {identity['checkpoint_step']}，完成 **{overall['episodes']} 次**；无异常碰撞的目标按压成功 **{overall['successes']}/{overall['episodes']}（{overall['success_rate']:.1%}）**。", "",
             "四个 worker 的独立审计均通过。任务成功率按实际按钮行程、按压杆接触力、误按和碰撞记录计算，单独列示。", "",
             f"12 个楼层各评估中心与四个角点，共 60 个不同条件；每个条件一次。前后 X 为 ±10 mm，左右 Y 为 ±25 mm。每次最多 {identity['max_sim_seconds']:g} 秒，30 Hz 动作、120 Hz 物理步进、3 点因果关节均值平滑。", "",
             "模型使用实时双相机、实测基座坐标系 TCP 状态和任务文字。所有原始模型动作按现有控制规则执行，未使用专家轨迹或按钮坐标驱动控制器。仿真在等待远程推理时暂停。", "",
             "| 面板位置 | 成功 / 次数 | 成功率 |", "|---|---:|---:|"]
    labels = {"center": "中心", "xmin_ymin": "X− / Y−", "xmin_ymax": "X− / Y+", "xmax_ymin": "X+ / Y−", "xmax_ymax": "X+ / Y+"}
    for name, stats in report["by_position"].items():
        lines.append(f"| {labels[name]} | {stats['successes']} / {stats['episodes']} | {stats['success_rate']:.1%} |")
    lines += ["", "| 楼层 | 成功 / 次数 |", "|---|---:|"]
    lines += [f"| {floor} | {stats['successes']} / {stats['episodes']} |" for floor, stats in report["by_floor"].items()]
    lines += ["", "终止原因：" + "、".join(f"{REASONS.get(k, k)} {v} 次" for k, v in overall["termination_counts"].items()) + "。", "",
              "图像审计核对输入哈希、视频帧数和时间戳；没有自动识别图像语义或认证橙灯在每个视角中可见。面板投影位于全局相机画幅内，也可能暂时被机械臂遮挡。每个条件只有一次试验，结果适用于本次固定评估条件。", "",
              "详细结果：[60 条 CSV](episodes.csv) · [聚合报告与原始审计哈希](aggregate_report.json)。CSV 保留每条全局和腕部原视频路径。", "",
              f"模型 SHA256：`{identity['checkpoint_sha256']}`。", ""]
    return "\n".join(lines)


def write_outputs(root, report):
    root = Path(root)
    destinations = [root / name for name in ("aggregate_report.json", "summary.md", "episodes.csv")]
    require(not any(p.exists() for p in destinations), "Aggregate outputs already exist; preserve the original evidence")
    stream = io.StringIO(newline="")
    rows = report["episodes"]
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})
    payloads = [json.dumps(report, indent=2, allow_nan=False) + "\n", summary_markdown(report), stream.getvalue()]
    for path, payload in zip(destinations, payloads):
        with path.open("x") as handle:
            handle.write(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-checkpoint-step", type=int, default=10600)
    args = parser.parse_args()
    report = aggregate(args.root, args.expected_checkpoint_sha256, args.expected_checkpoint_step)
    write_outputs(args.root, report)
    print(json.dumps({"aggregation_pass": report["aggregation_pass"], "all_worker_audits_pass": report["all_worker_audits_pass"],
                      "overall": report["overall"], "root": str(args.root.resolve())}, indent=2))


if __name__ == "__main__":
    main()
