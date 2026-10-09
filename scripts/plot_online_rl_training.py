#!/usr/bin/env python3
"""Reconstruct and plot online-RL training curves from immutable JSONL logs.

Example:
  .conda/envs/pressb/bin/python scripts/plot_online_rl_training.py \
      --run outputs/online_rl_fast_20261003

Only writes derived plots/CSVs; does not import Torch or connect to services.
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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np


METHODS = {
    "action_residual": ("Action residual", "#0072B2", "-"),
    "initial_noise": ("Initial noise", "#D55E00", "--"),
}
REASONS = ("target_pressed", "wrong_button_pressed", "time_limit", "unexpected_collision")
REASON_LABELS = ("Success", "Wrong button", "Timeout", "Collision")
REASON_COLORS = ("#009E73", "#E69F00", "#A6A6A6", "#CC79A7")


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def fingerprint(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": sha.hexdigest()}


def recent_mean(values, counts, window):
    """An exact trailing mean at each observed episode-prefix length."""
    values = np.asarray(values, dtype=float)
    counts = np.asarray(counts, dtype=int)
    prefix = np.r_[0., np.cumsum(values)]
    starts = np.maximum(0, counts - window)
    width = counts - starts
    return np.divide(prefix[counts] - prefix[starts], width,
                     out=np.full(len(counts), np.nan), where=width > 0)


def trailing_median(values, window=31):
    values = np.asarray(values, dtype=float)
    out = np.full_like(values, np.nan)
    for i, value in enumerate(values):
        # A missing logged value remains missing, even if past values exist.
        if np.isfinite(value):
            history = values[max(0, i + 1 - window):i + 1]
            out[i] = np.median(history[np.isfinite(history)])
    return out


def prepare(run, method, recent_window, throughput_window):
    folder = run / f"{method}_train"
    metrics = read_jsonl(folder / "metrics.jsonl")
    episodes = read_jsonl(folder / "episodes.jsonl")
    summary = read_json(folder / "summary.json")
    manifest = read_json(folder / "manifest.json")
    if len({row["episode_id"] for row in episodes}) != len(episodes):
        raise ValueError(f"{method}: duplicate episode IDs")
    if len(episodes) != summary["episodes"]:
        raise ValueError(f"{method}: incomplete episode log")
    success = np.array([bool(row["success"]) for row in episodes], dtype=float)
    duration = np.array([row["sim_seconds"] for row in episodes], dtype=float)
    reasons = np.array([row["termination"] for row in episodes])
    if not set(reasons) <= set(REASONS):
        raise ValueError(f"Unknown termination reasons: {set(reasons) - set(REASONS)}")
    prefix = np.r_[0, np.cumsum(success).astype(int)]
    for row in metrics + [summary]:
        if not 0 <= row["episodes"] <= len(episodes) or prefix[row["episodes"]] != row["successes"]:
            raise ValueError(f"{method}: episode prefix does not match metrics")
    for field in ("transitions", "episodes", "successes", "wall_seconds"):
        if np.any(np.diff([row[field] for row in metrics + [summary]]) < 0):
            raise ValueError(f"{method}: non-monotone {field}")

    # The final summary includes checkpoint/cleanup time. For duplicate x values
    # use the last observation, while retaining all raw loss samples separately.
    points = {row["transitions"]: row for row in metrics + [summary]}
    points = [points[key] for key in sorted(points)]
    x = np.array([row["transitions"] for row in points], dtype=int)
    t = np.array([row["wall_seconds"] for row in points], dtype=float)
    n = np.array([row["episodes"] for row in points], dtype=int)
    data = {
        "transitions": x, "wall_hours": t / 3600., "completed_episodes": n,
        "successes": prefix[n], "draining": np.array([row["draining"] for row in points]),
        "recent_window_count": np.minimum(n, recent_window),
        "recent_sr_100": recent_mean(success, n, 100),
        "recent_sr": recent_mean(success, n, recent_window),
        "cumulative_sr": np.divide(prefix[n], n, out=np.full(len(n), np.nan), where=n > 0),
        "recent_duration_seconds": recent_mean(duration, n, recent_window),
        "cumulative_throughput": x / t,
    }
    for reason in REASONS:
        data[f"recent_{reason}_fraction"] = recent_mean(reasons == reason, n, recent_window)
    if not np.allclose(sum(data[f"recent_{reason}_fraction"] for reason in REASONS), 1., equal_nan=True):
        raise ValueError("Termination fractions do not sum to one")

    # No extrapolation: anchor the denominator to an actual earlier log point
    # at least throughput_window transitions back. Keep the exact span in CSV.
    rates = np.full(len(x), np.nan)
    spans = np.zeros(len(x), dtype=int)
    wall_spans = np.full(len(x), np.nan)
    for i in range(len(x)):
        j = np.searchsorted(x, x[i] - throughput_window, side="right") - 1
        if j >= 0 and t[i] > t[j]:
            spans[i] = x[i] - x[j]
            wall_spans[i] = t[i] - t[j]
            rates[i] = spans[i] / wall_spans[i]
    data["recent_throughput"] = rates
    data["throughput_window_transitions"] = spans
    data["throughput_window_seconds"] = wall_spans
    return {"method": method, "data": data, "metrics": metrics, "summary": summary,
            "manifest": manifest, "episode_success": success, "episode_duration": duration}


def configure_style():
    # Small-multiple line grammar and Okabe-Ito palette adapted from the
    # ccf-visual-composer plotting recipes; Matplotlib preserves vector output.
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.titleweight": "semibold", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#7A8288", "axes.labelcolor": "#25323B",
        "text.color": "#25323B", "xtick.color": "#4E5A66", "ytick.color": "#4E5A66",
        "axes.grid": True, "grid.color": "#DFE4E8", "grid.linewidth": .6,
        "grid.alpha": .8, "axes.axisbelow": True,
        "legend.frameon": False, "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


def legend_handles():
    return [Line2D([0], [0], color=color, lw=2.4, ls=style, label=label)
            for label, color, style in METHODS.values()]


def common_figure(title, subtitle, nrows, ncols, size):
    fig, axes = plt.subplots(nrows, ncols, figsize=size, squeeze=False)
    fig.suptitle(title, fontsize=18, fontweight="semibold", x=.065, y=.98, ha="left", va="top")
    fig.text(.065, .98 - .34 / size[1], subtitle, fontsize=10, color="#586773", va="top")
    fig.legend(handles=legend_handles(), loc="upper right", bbox_to_anchor=(.985, .994),
               ncol=2, fontsize=10, handlelength=2.8)
    return fig, axes


def finish(fig, output, name, note, *, top=.85, bottom=.13, hspace=.43, left=.075):
    fig.subplots_adjust(left=left, right=.977, bottom=bottom, top=top, wspace=.26, hspace=hspace)
    fig.text(left, .032, note, fontsize=8.3, color="#586773", va="bottom", linespacing=1.45)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"{name}.{suffix}", dpi=190)
    plt.close(fig)
    return fig


def transition_axis(ax):
    ax.set_xlabel("Collected transitions (million)")
    ax.set_xlim(0, 1.02)
    ax.set_xticks(np.arange(0, 1.01, .2))


def sr_figure(runs, output, window):
    fig, axes = common_figure("Recent success rate", "Training episodes | full run | single training seed per method",
                              1, 2, (13.2, 5.5))
    for item in runs:
        label, color, style = METHODS[item["method"]]
        d = item["data"]
        for ax, x in zip(axes[0], (d["transitions"] / 1e6, d["wall_hours"])):
            ax.plot(x, d["recent_sr_100"] * 100, color=color, lw=.65, alpha=.18)
            ax.plot(x, d["recent_sr"] * 100, color=color, lw=2., ls=style)
        end = d["recent_sr"][-1] * 100
        axes[0, 1].scatter(d["wall_hours"][-1], end, color=color, s=24, zorder=4)
        axes[0, 1].annotate(f"{end:.1f}%", (d["wall_hours"][-1], end), xytext=(0, 10),
                            textcoords="offset points", ha="center", color=color, fontsize=10)
    axes[0, 0].set_title("(a) By collected transitions", loc="left", pad=12)
    axes[0, 1].set_title("(b) By elapsed training time", loc="left", pad=12)
    transition_axis(axes[0, 0])
    axes[0, 1].set_xlabel("Wall time since each method started (hours)")
    axes[0, 1].set_xlim(0, 21)
    axes[0, 1].set_xticks(np.arange(0, 21, 4))
    for ax in axes[0]:
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.set_ylabel("Recent success rate")
    return finish(fig, output, "recent_sr",
           f"Bold: trailing {window:,} completed episodes; faint: trailing 100. Early windows use all completed episodes available.\n"
           "Training SR is separate from fixed-condition evaluation. Endpoints include draining the last active episodes.",
           top=.81, bottom=.21)


def overview_figure(runs, output, window, throughput_window):
    fig, axes = common_figure("Training progress", "Success accumulation, task duration, collection throughput and episode count",
                              2, 2, (13.2, 8.5))
    fields = ("cumulative_sr", "recent_duration_seconds", "recent_throughput", "completed_episodes")
    titles = ("(a) Cumulative training success rate", f"(b) Mean duration of recent {window:,} episodes",
              f"(c) Throughput over ~{throughput_window / 1000:g}k transitions", "(d) Completed training episodes")
    labels = ("Cumulative success rate", "Simulated seconds per episode", "Transitions / wall-clock second", "Completed episodes (thousand)")
    for item in runs:
        _, color, style = METHODS[item["method"]]
        d = item["data"]
        for ax, field in zip(axes.flat, fields):
            scale = 100 if field == "cumulative_sr" else .001 if field == "completed_episodes" else 1.
            ax.plot(d["transitions"] / 1e6, d[field] * scale, color=color, ls=style, lw=1.9)
    for ax, title, label in zip(axes.flat, titles, labels):
        ax.set_title(title, loc="left", pad=10)
        ax.set_ylabel(label)
        transition_axis(ax)
        ax.set_ylim(bottom=0)
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(100))
    return finish(fig, output, "training_overview",
           "Duration is simulation time; throughput includes inference, simulation, resets, updates and checkpoint pauses.\n"
           "Throughput uses observed log endpoints, with exact window spans in CSV. RPC and SAC timings overlap.",
           top=.85, bottom=.13)


def losses_figure(runs, output):
    fig, axes = common_figure("SAC optimization diagnostics", "Logged minibatch snapshots | not averages over all training updates",
                              2, 3, (14.5, 8.4))
    specs = (
        ("critic_loss", "(a) Critic loss", "Critic loss", "log"),
        ("actor_loss", "(b) Actor loss", "Actor loss (symlog)", "symlog"),
        ("alpha", "(c) Entropy temperature", "Alpha", "log"),
        ("actor_entropy", "(d) Actor entropy", "Logged actor entropy", "linear"),
        ("critic_grad_norm", "(e) Critic gradient norm", "Gradient norm", "log"),
        ("actor_grad_norm", "(f) Actor gradient norm", "Gradient norm", "log"),
    )
    for item in runs:
        _, color, style = METHODS[item["method"]]
        records = item["metrics"]
        x = np.array([row["transitions"] for row in records]) / 1e6
        for ax, (field, _, _, scale) in zip(axes.flat, specs):
            values = np.array([row.get(field, np.nan) for row in records], dtype=float)
            if scale == "log":
                values[values <= 0] = np.nan
            ax.plot(x, values, color=color, lw=.45, alpha=.20)
            ax.plot(x, trailing_median(values), color=color, lw=1.6, ls=style)
    for ax, (_, title, label, scale) in zip(axes.flat, specs):
        ax.set_title(title, loc="left", pad=10)
        ax.set_ylabel(label)
        transition_axis(ax)
        if scale == "symlog":
            ax.set_yscale("symlog", linthresh=.1)
        else:
            ax.set_yscale(scale)
    return finish(fig, output, "optimization_diagnostics",
           "Bold: trailing median of 31 logged values; faint: original logged values. Missing values remain missing.\n"
           "Action spaces and entropy definitions differ between methods; loss magnitudes are diagnostic, not policy rankings.",
           top=.85, bottom=.14, hspace=.44)


def termination_figure(runs, output, window):
    fig, axes = common_figure("How episodes end", f"Trailing {window:,} completed episodes | fractions sum to 100%",
                              1, 2, (13.2, 5.7))
    for leg in fig.legends:
        leg.remove()
    for ax, item in zip(axes[0], runs):
        d = item["data"]
        ax.stackplot(d["transitions"] / 1e6,
                     [d[f"recent_{reason}_fraction"] * 100 for reason in REASONS],
                     labels=REASON_LABELS, colors=REASON_COLORS, alpha=.9, linewidth=0)
        ax.set_title(METHODS[item["method"]][0], loc="left", pad=12)
        transition_axis(ax)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Fraction of recent episodes")
        ax.yaxis.set_major_formatter(PercentFormatter(100))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, .89), ncol=4)
    return finish(fig, output, "termination_breakdown",
           "Counts follow episode log order; simultaneous batch completions are logged by environment ID.\n"
           "Success and failure terminate episodes immediately; the simulation timeout remains 15 seconds.",
           top=.77, bottom=.20)


def eval_figure(run, output):
    names = ("base", "action_residual", "initial_noise")
    labels = ("Frozen VLA-JEPA", "Action residual", "Initial noise")
    summaries = [read_json(run / f"{name}_eval" / "summary.json") for name in names]
    rates = [row["success_rate"] * 100 for row in summaries]
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    fig.suptitle("Final fixed-condition evaluation", fontsize=17, fontweight="semibold", x=.1, y=.97, ha="left")
    fig.text(.1, .91, "12 floors × 5 panel layouts × 2 repetitions = 120 episodes per policy", fontsize=10, color="#586773")
    bars = ax.bar(labels, rates, color=("#8B96A3", "#0072B2", "#D55E00"), width=.56)
    for bar, row in zip(bars, summaries):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{row['success_rate'] * 100:.1f}%\n{row['successes']}/{row['episodes']}",
                ha="center", va="bottom", fontsize=12, linespacing=1.4)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Evaluation success rate")
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.grid(axis="x", visible=False)
    return finish(fig, output, "final_evaluation",
           "Single trained policy per method; no intermediate evaluations were logged.\n"
           "Residual uses a deterministic actor; initial noise uses reproducible stochastic sampling.",
           top=.80, bottom=.23, left=.12)


def write_csv(output, item):
    data = item["data"]
    with (output / f"{item['method']}_curves.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(data.keys())
        for values in zip(*data.values()):
            writer.writerow(["" if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v for v in values])


def write_notes(output, metadata, window, throughput_window):
    lines = [
        "# 在线 RL 训练曲线", "", "全部图由现有日志离线生成，没有重新训练或补造评估点。", "",
        "## 图像", "",
        "- `recent_sr`：最近成功率，按采集 transition 与各方法独立墙钟时间展示。",
        "- `training_overview`：累计成功率、recent episode 仿真时长、近期吞吐、累计 episode 数。",
        "- `optimization_diagnostics`：critic/actor loss、alpha、entropy、梯度范数。",
        "- `termination_breakdown`：成功、误按、超时、碰撞的近期比例。",
        "- `final_evaluation`：最终120条固定条件评估，与训练 SR 分开。",
        "", "每张图提供 PNG、可编辑 SVG 与矢量 PDF。`training_curves.pdf` 合并全部5张图。", "", "## 统计口径", "",
        f"主曲线 recent SR = 最近 {window} 个已完成并写入日志的 episode 的成功比例；不足时使用全部已完成 episode。浅线为最近100条。",
        "在每个 metrics 记录处，取 `episodes=N`，用 `episodes.jsonl` 的前N条作精确前缀。已逐行核对其成功数与 metrics.successes 一致。",
        "同一批内 episode 按 env_id 写入，窗口不声称按每个物理终止 tick 排序。首个metrics采样位置见清单，不伪造 x=0 观测。",
        f"吞吐用相隔至少 {throughput_window} transition 的真实日志端点之差计算，CSV保存实际分母；末点包含最终检查点和收尾耗时。",
        "episode时长单位是仿真秒，横轴wall time是各自训练从启动到结束的实际小时，两种方法在机器上按顺序训练。",
        "loss图浅线为原始minibatch日志点，粗线为最近31个已记录值的中位数，不代表全部更新的平均loss。缺失actor指标不填零。",
        "", "## 解读边界", "",
        "- 每种方法只有一个训练seed；浅线表示短窗口/原始波动，不是多seed置信区间。",
        "- 达到1M后停止补充episode，排空最后64个活跃环境。短窗口末端下跌受收尾样本构成影响，不能直接认定策略崩溃。",
        "- 动作残差从旧10207 transition实验仅迁移actor；图中x轴仅计本次新训练，critic/replay重新初始化。",
        "- 动作残差前2000条不启用残差，之后启用概率在50000条前逐步升至1；初始噪声方法使用自己的warmup和探索配置。",
        "- 随机训练采样SR与最终固定条件评估SR不同；没有中途checkpoint评估曲线，不能把训练recent SR标为eval SR。",
        "- 残差最终评估actor确定性，噪声最终评估按固定条件seed随机采样；两者任务条件与冻结模型相同。",
        "", "## 最后记录", "",
        f"| 方法 | recent{window}（最后非draining） | recent{window}（含收尾） | recent100（含收尾） |",
        "|---|---:|---:|---:|",
    ]
    for name, row in metadata["series"].items():
        lines.append(f"| {METHODS[name][0]} | {row['last_active_recent_sr']:.1%} | {row['final_recent_sr']:.1%} | {row['final_recent_sr_100']:.1%} |")
    lines += ["", "## 复现", "", "```bash",
              f".conda/envs/pressb/bin/python scripts/plot_online_rl_training.py --run outputs/{metadata['run_name']}",
              "```", "", "`plot_manifest.json` 保存窗口定义、日志SHA-256、绘图脚本SHA-256和核对结果。",
              "每个方法的 `*_curves.csv` 保存精确导出曲线，原始loss仍可由源metrics复现。", ""]
    (output / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--recent-episodes", type=int, default=1000)
    parser.add_argument("--throughput-transitions", type=int, default=10000)
    args = parser.parse_args()
    if args.recent_episodes < 1 or args.throughput_transitions < 1:
        parser.error("window sizes must be positive")
    run = args.run.resolve()
    output = (args.output or run / "plots").resolve()
    if output == run or output in [run / f"{m}_train" for m in METHODS]:
        parser.error("output must be a separate derived-artifact directory")
    output.mkdir(parents=True, exist_ok=True)
    contract = {
        "artifact": "offline training diagnostics", "destination": "research analysis; no venue-specific claim",
        "question": "How did training SR, task duration, throughput and optimization evolve?",
        "source": "unchanged local metrics.jsonl, episodes.jsonl, summary.json, manifest.json",
        "grammar": "small-multiple line plots; separate final evaluation bars",
        "statistics": "exact episode-prefix trailing windows; single training seed; no uncertainty band",
        "palette": "Okabe-Ito blue/vermillion plus solid/dashed method lines",
        "formats": ["png", "pdf", "svg", "csv"],
    }
    (output / "visual_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    runs = [prepare(run, method, args.recent_episodes, args.throughput_transitions) for method in METHODS]
    configure_style()
    figures = [
        sr_figure(runs, output, args.recent_episodes),
        overview_figure(runs, output, args.recent_episodes, args.throughput_transitions),
        losses_figure(runs, output),
        termination_figure(runs, output, args.recent_episodes),
        eval_figure(run, output),
    ]
    with PdfPages(output / "training_curves.pdf") as report:
        report.infodict()["Title"] = "Online RL training curves"
        for figure in figures:
            report.savefig(figure)
    metadata = {"run_name": run.name, "recent_episodes": args.recent_episodes,
                "short_recent_episodes": 100, "throughput_transitions": args.throughput_transitions,
                "loss_trailing_logged_values": 31, "source_files": {}, "series": {},
                "script_sha256": fingerprint(Path(__file__))["sha256"], "visual_contract": contract}
    for item in runs:
        method, data = item["method"], item["data"]
        active = np.flatnonzero(~data["draining"])[-1]
        metadata["series"][method] = {
            "metric_rows": len(item["metrics"]), "curve_rows": len(data["transitions"]),
            "episodes": int(item["summary"]["episodes"]), "prefix_count_checks_passed": True,
            "first_logged_transition": int(data["transitions"][0]),
            "final_transition": int(data["transitions"][-1]),
            "last_active_transition": int(data["transitions"][active]),
            "last_active_recent_sr": float(data["recent_sr"][active]),
            "final_recent_sr": float(data["recent_sr"][-1]),
            "final_recent_sr_100": float(data["recent_sr_100"][-1]),
        }
        write_csv(output, item)
        for filename in ("metrics.jsonl", "episodes.jsonl", "summary.json", "manifest.json"):
            relative = f"{method}_train/{filename}"
            metadata["source_files"][relative] = fingerprint(run / relative)
    for name in ("base", *METHODS):
        for filename in ("summary.json", "episodes.jsonl", "manifest.json"):
            relative = f"{name}_eval/{filename}"
            metadata["source_files"][relative] = fingerprint(run / relative)
    (output / "plot_manifest.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    write_notes(output, metadata, args.recent_episodes, args.throughput_transitions)
    print(json.dumps({"output": str(output), "series": metadata["series"]}, indent=2))


if __name__ == "__main__":
    main()
