#!/usr/bin/env python3
"""Create a readable report and plots from an audited closed-loop policy run."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audit_policy_eval import (PANEL_FIELDS, episode_schedule, outcome_statistics, panel_episode_geometry,
                               panel_layout_plan, policy_identity, require, sha, smoothing_settings)
from diagnose_policy_eval import motion_smoothness


def summary_identity(root, audit):
    manifest = json.loads((root / "eval_manifest.json").read_text())
    health = json.loads((root / "policy_service.json").read_text())
    identity = policy_identity(manifest, health)
    audited = audit["provenance"]
    require(audited["checkpoint_sha256"] == identity["checkpoint_sha256"]
            and audited["eval_manifest_sha256"] == sha(root / "eval_manifest.json"),
            "Summary model/manifest differs from the independently audited run")
    if "checkpoint_step" in audited:
        require(audited["checkpoint_step"] == identity["checkpoint_step"], "Summary checkpoint step differs from audit")
    if "policy_service_sha256" in audited:
        require(audited["policy_service_sha256"] == sha(root / "policy_service.json"), "Policy service record changed after audit")
    return identity, health, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    report = json.loads((root / "report.json").read_text())
    if not report.get("complete"):
        raise ValueError("Evaluation must be complete")
    audit = json.loads((root / "audit.json").read_text())
    if (audit.get("audit_pass") is not True or audit.get("audited_episodes") != report["total_episodes"]
            or audit.get("collision_free_successes") != report["passed_episodes"]):
        raise ValueError("Independent audit must pass and match the complete run")
    identity, health, manifest = summary_identity(root, audit)
    layouts = panel_layout_plan(manifest)
    schedule = episode_schedule(manifest)
    require(sorted(row['episode_id'] for row in report['episodes']) == list(range(len(schedule))),
            'Summary requires the complete requested floor/repeat/layout schedule')
    collection = json.loads((Path(manifest['arguments']['dataset']) / 'meta/collection_metadata.json').read_text())
    audited_rows = {row['episode']: row for row in audit['episodes']}
    smoothing = smoothing_settings(manifest)
    if 'motion_smoothing' in audit:
        require(audit['motion_smoothing'] == smoothing, 'Summary smoothing differs from audit')
    output = root / "summary"
    output.mkdir(exist_ok=False)
    rows = []
    figure, axes = plt.subplots(len(layouts), 2, figsize=(13, 4.5 * len(layouts)), squeeze=False)
    for episode in report["episodes"]:
        directory = root / f"episode_{episode['episode_id']:06d}"
        meta = json.loads((directory / "metadata.json").read_text())
        evidence = audited_rows[directory.name]
        require(all(sha(directory / name) == digest for name, digest in evidence['files_sha256'].items()),
                'Episode source changed after independent audit')
        geometry = panel_episode_geometry(meta, collection)
        require(meta['success'] == evidence['success'] and meta['task_success'] == evidence['task_success'],
                'Episode outcome differs from the independent audit')
        require(smoothing_settings(meta) == smoothing, 'Episode smoothing differs from manifest')
        with np.load(directory / "physics.npz", allow_pickle=False) as f:
            distance = f["target_tip_distance_m"].copy()
            state = f["state"].copy()
            time = f["sim_time"].copy()
            force = f["contact_force"].copy()
            travel = f["button_travel"].copy()
            tracking = np.max(np.abs(f["q_actual"] - f["q_command"]), axis=1)
            smoothness = motion_smoothness({key: f[key] for key in ('state', 'q_command', 'q_command_unsmoothed') if key in f})
        requests = [json.loads(line) for line in (directory / "requests.jsonl").read_text().splitlines()]
        actions = [json.loads(line) for line in (directory / "actions.jsonl").read_text().splitlines()]
        row = dict(meta, initial_target_tip_distance_m=float(distance[0]),
                   panel_geometry=geometry,
                   max_tcp_displacement_from_start_m=float(np.linalg.norm(state[:, :3]-state[0, :3], axis=1).max()),
                   max_contact_force_n=float(force.max()), max_button_travel_m=float(travel.max()),
                   mean_joint_tracking_error_rad=float(tracking.mean()),
                   motion_smoothness=smoothness,
                   mean_command_position_residual_m=float(np.mean([a["command_position_residual_m"] for a in actions])) if actions else None,
                   mean_executed_position_residual_m=float(np.mean([a['executed_position_residual_m'] for a in actions]))
                       if actions and all('executed_position_residual_m' in a for a in actions) else None,
                   mean_rpc_seconds=float(np.mean([r["wall_seconds"] for r in requests])) if requests else None)
        rows.append(row)
        row.update({key: geometry[key] for key in PANEL_FIELDS})
        left, right = axes[row['panel_layout_index']]
        label = str(meta['floor']) + (f" / repeat {meta['repeat']}" if manifest['arguments']['episodes_per_floor'] > 1 else '')
        color = plt.get_cmap('tab20')((meta['floor'] - 24) / 12)
        left.plot(time, distance * 1000, label=label, color=color, linewidth=1.2)
        right.plot(time, np.linalg.norm(state[:, :3]-state[0, :3], axis=1)*1000, color=color, linewidth=1.2)
    for layout, (left, right) in zip(layouts, axes):
        title = f"{layout['panel_layout_name']} (X={layout['panel_offset_x_m']*1000:g}, Y={layout['panel_offset_y_m']*1000:g} mm)"
        left.set(title=title + " — tool tip distance", xlabel="Simulation time (s)", ylabel="Distance (mm)")
        right.set(title=title + " — TCP displacement", xlabel="Simulation time (s)", ylabel="Displacement (mm)")
        left.legend(title="Floor", ncol=4, fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=.2)
    figure.suptitle(f"VLA-JEPA step {health['checkpoint_step']}: {report['passed_episodes']}/{report['total_episodes']} successful presses")
    figure.tight_layout()
    figure.savefig(output / "motion.png", dpi=160)
    plt.close(figure)
    reasons = dict(Counter(row["termination"] for row in rows))
    outcomes = outcome_statistics(rows)
    require(outcomes['overall']['successes'] == report['passed_episodes'], 'Aggregated success count differs from report')
    summary = dict(**identity, model_sha256=health["model_sha256"],
                   panel_layout_mode=manifest.get('panel_layout_mode', 'fixed'), panel_layouts=layouts, outcomes=outcomes,
                   motion_smoothing=smoothing, total_episodes=len(rows), success_count=report["passed_episodes"], termination_counts=reasons, episodes=rows)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    source_paths = [root / "report.json", root / "audit.json", root / "policy_service.json", root / "eval_manifest.json", Path(__file__)]
    (output / "sources.json").write_text(json.dumps({str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                                      for path in source_paths}, indent=2) + "\n")
    lines = ["# 当前权重的 Isaac Sim 闭环评估", "",
             f"第 {health['checkpoint_step']} 步权重；成功 **{report['passed_episodes']}/{len(rows)}**。", "",
             f"模型输入为实时全局相机、腕部相机、实测基座坐标系夹爪 TCP 状态和任务文字。每次预测 {health['action_horizon']} 个绝对目标，按 {health['fps']} Hz 执行、{1 / manifest['config']['physics_dt']:g} Hz 物理步进。等待 H200 推理时暂停仿真时间。夹爪固定关闭。", "",
             f"成功要求按压杆与目标按键接触力超过 0.02 N、按钮行程至少 {manifest['config']['press_threshold'] * 1000:g} mm，同时没有误按或异常碰撞；不执行撤回。共 {len(layouts)} 个面板位置，每个位置的每个所选楼层 {manifest['arguments']['episodes_per_floor']} 次。本结果不代表充分统计的泛化成功率。", "",
             "原始 XYZ 不缩放、不改坐标系；6D 旋转按训练约定转换为四元数。逆运动学受关节位置和速度限制，控制误差单独记录。未使用录制动作、目标轨迹或目标按键坐标驱动模型/控制器。", "",
             f"关节目标先在 120 Hz 线性插值，再使用 {smoothing['window']} 点因果移动均值；标称延迟 {smoothing['nominal_delay_s'] * 1000:.2f} ms。历史跨预测窗口保留，每条 episode 以初始命令重新填充；窗口 1 等同于不平滑。IK 残差对应原解算目标，执行残差对应实际下发的命令，末段不足四个物理步时按实际终止时刻计算。", "",
             "| 位置 | X/Y 偏移 (mm) | 成功 / 总次数 |", "|---|---:|---:|"]
    for layout in layouts:
        count = outcomes['by_position'][layout['panel_layout_name']]
        lines.append(f"| {layout['panel_layout_name']} | {layout['panel_offset_x_m']*1000:g} / {layout['panel_offset_y_m']*1000:g} | {count['successes']} / {count['episodes']} |")
    names = [layout['panel_layout_name'] for layout in layouts]
    lines += ['', '| 楼层 | ' + ' | '.join(names) + ' | 合计 |', '|---|' + '---:|' * (len(names) + 1)]
    for floor, count in outcomes['by_floor'].items():
        cells = [outcomes['by_floor_position'][floor][name] for name in names] + [count]
        lines.append(f"| {floor} | " + ' | '.join(f"{c['successes']} / {c['episodes']}" for c in cells) + ' |')
    lines += ['', '| Episode / 位置 / repeat | 楼层 | 终止原因 | 仿真秒数 | 最近工具尖端距离 (mm) | 最大 TCP 移动 (mm) | 视频 |',
              '|---|---|---|---:|---:|---:|---|']
    labels = dict(time_limit="超时", target_pressed="目标按亮", wrong_button_pressed="误按", unexpected_collision="异常碰撞", invalid_policy_action="无效预测")
    for row in rows:
        name = f"episode_{row['episode_id']:06d}"
        reason = labels.get(row['termination'], row['termination'])
        if row['termination'] == 'wrong_button_pressed':
            reason += " " + ", ".join(str(event['floor']) for event in row['events'] if event['type'] == 'pressed') + " 层"
        lines.append(f"| {row['episode_id']} / {row['panel_layout_name']} / {row['repeat']} | {row['floor']} | {reason} | {row['sim_seconds']:.3f} | {row['min_target_tip_distance_m']*1000:.1f} | {row['max_tcp_displacement_from_start_m']*1000:.1f} | [全局](../{name}/global.mp4) · [腕部](../{name}/wrist.mp4) |")
    lines += ["", "| Episode / 位置 / repeat | 楼层 | 平均 IK 位置残差 (mm) | 平均执行位置残差 (mm) | 关节 jerk RMS，平滑前 → 后 (rad/s³) | 实际 TCP jerk RMS (m/s³) |",
              "|---|---|---:|---:|---:|---:|"]
    def readable(value, scale=1.):
        return '—' if value is None else f'{value * scale:.2f}'
    for row in rows:
        metrics = row['motion_smoothness']
        before = metrics['unsmoothed_joint_command']['jerk']['rms']
        after = metrics['executed_joint_command']['jerk']['rms']
        measured = metrics['actual_tcp_position']['jerk']['rms']
        lines.append(f"| {row['episode_id']} / {row['panel_layout_name']} / {row['repeat']} | {row['floor']} | {readable(row['mean_command_position_residual_m'], 1000)} | {readable(row['mean_executed_position_residual_m'], 1000)} | {readable(before)} → {readable(after)} | {readable(measured)} |")
    lines += ["", "jerk 由同一条记录的 120 Hz 样本直接三阶差分计算，不补帧；RMS 汇总全部时间与坐标分量。不同闭环运行的轨迹会变化，跨运行指标不能单独证明平滑的因果效果。", "",
              "![实测运动](motion.png)", "", "工具尖端距离只是诊断指标，任务成功由实际接触力和按钮行程判断。", "",
              "独立审计检查输入图像哈希、视频尺寸/帧数/时间戳、动作转换、控制时序与物理记录，不自动识别图像中的橙色亮灯。面板投影证明完整面板处于相机视锥内，不保证无遮挡；审计通过也不表示策略完成了任务。", "",
              "完整证据：[执行报告](../report.json)、[独立审计](../audit.json)、[模型身份](../policy_service.json)、[采集与控制约定](../eval_manifest.json)、[派生诊断](metrics.json)。", "",
              f"Checkpoint：`{identity['checkpoint']}`。", "",
              f"模型 SHA256：`{health['model_sha256']}`。", ""]
    (output / "README.md").write_text("\n".join(lines))
    print(json.dumps(dict(output=str(output), success_count=report["passed_episodes"], total_episodes=len(rows), termination_counts=reasons)))


if __name__ == "__main__":
    main()
