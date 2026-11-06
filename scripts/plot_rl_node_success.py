#!/usr/bin/env python3
"""Plot audited success curves from configurable RL-node logs, without GPU/RPC.

Example:
  python3 scripts/plot_rl_node_success.py \
    --run outputs/pap_dsrl_local_400k_20261105/train
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np


def load_episodes(path):
    transitions, outcomes, ids = [], [], set()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            if not raw.strip():
                continue
            row = json.loads(raw)
            if row["episode_id"] in ids or type(row["success"]) is not bool:
                raise ValueError("Duplicate episode ID or invalid success outcome")
            ids.add(row["episode_id"])
            transitions.append(row["transition"])
            outcomes.append(row["success"])
    x, success = np.asarray(transitions, dtype=np.int64), np.asarray(outcomes, dtype=np.int64)
    if not len(x) or np.any(np.diff(x) < 0):
        raise ValueError("Empty or nonmonotonic episode log")
    return x, success, digest.hexdigest()


def trailing(prefix, counts, width, *, full_window):
    start = np.maximum(0, counts - width)
    span = counts - start
    valid = counts >= width if full_window else span > 0
    return np.divide(prefix[counts] - prefix[start], span,
                     out=np.full(len(counts), np.nan), where=valid)


def write_csv(path, header, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for row in rows:
            writer.writerow(["" if isinstance(v, (float, np.floating)) and not np.isfinite(v)
                             else v for v in row])


def render(run, output, title):
    config = json.loads((run / "config.json").read_text())
    status = json.loads((run / "status.json").read_text())
    if status["state"] != "completed":
        raise ValueError("Use a completed run so the input files remain stable")
    if config.get("checkpoint", {}).get("resume"):
        raise ValueError("This plot requires a complete fresh-run episode log")
    if config["rollout"].get("recent_window", 100) != 100:
        raise ValueError("Expected logged recent_window=100")
    episode_x, outcomes, episode_sha = load_episodes(run / "episodes.jsonl")
    prefix = np.r_[0, np.cumsum(outcomes)]
    raw_metrics = (run / "metrics.jsonl").read_bytes()
    metrics = [json.loads(line) for line in raw_metrics.splitlines() if line.strip()]
    if not metrics:
        raise ValueError("Empty metrics log")
    if any(b["transitions"] < a["transitions"] or b["elapsed_seconds"] < a["elapsed_seconds"]
           for a, b in zip(metrics, metrics[1:])):
        raise ValueError("Nonmonotonic metric log")
    for row in metrics + [status]:
        count = int(np.searchsorted(episode_x, row["transitions"], side="right"))
        if count != row["episodes"] or int(prefix[count]) != row["successes"]:
            raise ValueError("Episode-prefix counts disagree with metrics/status")
        expected = trailing(prefix, np.array([count]), 100, full_window=False)[0]
        actual = row.get("recent_sr")
        if count == 0:
            if actual is not None:
                raise ValueError("SR must be absent before any episode finishes")
        elif actual is None or not np.isclose(actual, expected, rtol=0, atol=1e-12):
            raise ValueError("Logged recent SR disagrees with episode outcomes")
        cumulative = int(prefix[count]) / count if count else None
        if cumulative is not None and not np.isclose(row["success_rate"], cumulative, rtol=0, atol=1e-12):
            raise ValueError("Logged cumulative SR disagrees with episode outcomes")
    if len(outcomes) != status["episodes"]:
        raise ValueError("Episode log extends beyond final status")

    # Plot at actual metric boundaries, replacing the duplicate last point with
    # completed status so the final elapsed time includes checkpoint saving.
    rows = {row["transitions"]: row for row in metrics + [status]}
    rows = [rows[key] for key in sorted(rows)]
    x = np.asarray([row["transitions"] for row in rows])
    counts = np.asarray([row["episodes"] for row in rows])
    hours = np.asarray([row["elapsed_seconds"] for row in rows]) / 3600
    recent100 = trailing(prefix, counts, 100, full_window=True)
    recent1000 = trailing(prefix, counts, 1000, full_window=True)
    cumulative = np.divide(prefix[counts], counts, out=np.full(len(counts), np.nan), where=counts > 0)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "success_rate_training.csv",
              ["transitions", "elapsed_hours", "completed_episodes", "successes",
               "recent_sr_100", "recent_sr_1000", "cumulative_sr"],
              zip(x, hours, counts, prefix[counts], recent100, recent1000, cumulative))
    episode_counts = np.arange(1, len(outcomes) + 1)
    write_csv(output / "success_rate_by_episode.csv",
              ["completion_transition", "completed_episodes", "success", "recent_sr_100",
               "recent_sr_1000", "cumulative_sr"],
              zip(episode_x, episode_counts, outcomes,
                  trailing(prefix, episode_counts, 100, full_window=True),
                  trailing(prefix, episode_counts, 1000, full_window=True),
                  prefix[episode_counts] / episode_counts))

    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"font.size": 11, "axes.unicode_minus": False, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "figure.facecolor": "white",
                         "axes.facecolor": "white", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#B7C2CB",
                         "text.color": "#182D40", "axes.labelcolor": "#334B60",
                         "xtick.color": "#536778", "ytick.color": "#536778"})
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 8.3), sharex=True,
                             gridspec_kw={"height_ratios": [1.45, 1]})
    fig.subplots_adjust(left=.085, right=.965, top=.80, bottom=.145, hspace=.38)
    duration = round(status["elapsed_seconds"] / 60)
    fig.suptitle(title, x=.085, y=.965, ha="left", fontsize=22, fontweight="bold")
    fig.text(.085, .913,
             f"{x[-1] / 1000:g}k transitions  ·  {len(outcomes):,} 个完成 episode  ·  "
             f"用时 {duration // 60} 小时 {duration % 60} 分钟  ·  "
             f"{config['rollout']['num_envs']} 环境并行",
             color="#536778", fontsize=11)
    ax = axes[0]
    ax.plot(x / 1000, recent100 * 100, color="#68ADD4", linewidth=.7, alpha=.58,
            label=f"最近 100 个 episode（结束时 {recent100[-1]:.1%}）")
    ax.plot(x / 1000, recent1000 * 100, color="#086899", linewidth=2.1,
            label=f"最近 1,000 个 episode（结束时 {recent1000[-1]:.1%}）")
    ax.set_title("近期成功率", loc="left", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#E4EAF0",
              framealpha=.97, fontsize=10, handlelength=2.6)
    ax = axes[1]
    ax.plot(x / 1000, cumulative * 100, color="#C57A25", linewidth=2.1)
    ax.set_title("累计成功率", loc="left", fontsize=13, fontweight="bold", pad=12)
    ax.annotate(f"{cumulative[-1]:.2%}", (x[-1] / 1000, cumulative[-1] * 100),
                xytext=(-8, -25), textcoords="offset points", ha="right", va="top",
                fontsize=15, fontweight="bold", color="#A95E15")
    warmup = config["update"]["learning_starts"]
    residual_ramp = (config["update"].get("progressive_exploration_steps", 0)
                     if config["method"] == "action_residual" else 0)
    for ax in axes:
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.set_ylabel("成功率")
        ax.grid(axis="both", color="#E4EAF0", linewidth=.65)
        ax.set_axisbelow(True)
        ax.axvspan(0, warmup / 1000, color="#7A8791", alpha=.14, linewidth=0)
        ax.axvline(warmup / 1000, color="#7A8791", linestyle=":", linewidth=.8)
        if residual_ramp:
            ax.axvspan(warmup / 1000, (warmup + residual_ramp) / 1000,
                       color="#E8B659", alpha=.13, linewidth=0)
            ax.axvline((warmup + residual_ramp) / 1000, color="#BA9048",
                       linestyle=":", linewidth=.8)
        ax.set_xlim(0, x[-1] / 1000)
    if residual_ramp:
        axes[0].text((warmup + residual_ramp / 2) / 1000, 14,
                     "渐进启用残差", ha="center", fontsize=9, color="#947035")
    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))
    axes[1].set_xlabel("累计采集的 transition（千）", labelpad=10)
    phase_note = f"灰色区域为前 {warmup:,} 条 warmup。"
    if residual_ramp:
        phase_note = (f"灰色：前 {warmup / 1000:g}k 零残差；"
                      f"浅黄色：{warmup / 1000:g}k–{(warmup + residual_ramp) / 1000:g}k 渐进启用残差。")
    fig.text(.085, .065,
             "按已完成 episode 统计；滚动窗口满额后开始显示。" + phase_note,
             color="#647687", fontsize=9.5)
    fig.text(.085, .031,
             "这些是带探索的训练成功率，不是独立评测结果。1 条 transition 对应一个环境的一段动作 chunk。",
             color="#647687", fontsize=9.5)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"success_rate_training.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)

    summary = {"run_id": status["run_id"], "metric_records_checked": len(metrics),
               "episodes": len(outcomes), "successes": int(prefix[-1]),
               "transitions": int(x[-1]), "elapsed_seconds": status["elapsed_seconds"],
               "final_recent_sr_100": float(recent100[-1]),
               "final_recent_sr_1000": float(recent1000[-1]),
               "final_cumulative_sr": float(cumulative[-1]),
               "all_logged_counts_and_rates_match_episodes": True,
               "rolling_windows": "trailing completed episodes; omit incomplete windows",
               "drawn_points": "actual metric boundaries plus completed final status; no fitted curve",
               "warmup_transitions": warmup,
               "residual_ramp_transitions": residual_ramp,
               "sources": {"episodes_sha256": episode_sha,
                           "metrics_sha256": hashlib.sha256(raw_metrics).hexdigest(),
                           "run": str(run.resolve())}}
    (output / "success_rate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(output / "success_rate_training.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Directory containing completed status/metrics/episodes")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--title", default="PAP · Wall-OSS DSRL 训练成功率")
    args = parser.parse_args()
    render(args.run, args.output or args.run.parent / "plots", args.title)
