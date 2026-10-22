#!/usr/bin/env python3
"""Render actual PAP evaluation counts and Wilson intervals from summary.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.ticker import PercentFormatter


def draw_panel(axis, entries, title):
    labels = []
    for index, (name, color, stats) in enumerate(entries):
        completed, planned = stats["completed"], stats["planned"]
        successes = stats["successes"]
        labels.append(f"{name}\n成功 {successes} / 已评 {completed}\n计划 {planned}")
        rate, interval = stats["success_rate"], stats["wilson_95"]
        if completed == 0:
            if rate is not None or interval is not None:
                raise ValueError("Unstarted group must not contain a numeric success rate")
            axis.text(index, 22, "待评估", ha="center", va="center", color="#77818d", fontsize=12)
            continue
        if rate is None or interval is None:
            raise ValueError("Completed trials need a success rate and Wilson interval")
        if not np.isclose(rate, successes / completed):
            raise ValueError("Success rate does not match actual completed denominator")
        value, low, high = 100 * rate, 100 * interval[0], 100 * interval[1]
        axis.bar(index, value, width=.56, color=color, edgecolor="#465263", linewidth=.9, zorder=2)
        axis.errorbar(index, value, yerr=[[max(0., value-low)], [max(0., high-value)]],
                      fmt="none", ecolor="#263347", capsize=5, capthick=1.3, elinewidth=1.4, zorder=3)
        axis.text(index, min(high+3, 107), f"{value:.1f}%", ha="center", va="bottom",
                  fontsize=13, weight="bold", color="#172b43")
    axis.set_xticks(range(len(entries)), labels, fontsize=10, linespacing=1.6)
    axis.set_xlim(-.65, len(entries)-.35)
    axis.set_ylim(0, 114)
    axis.set_yticks([0, 20, 40, 60, 80, 100])
    axis.yaxis.set_major_formatter(PercentFormatter(100))
    axis.set_ylabel("任务成功率", fontsize=11)
    axis.set_title(title, fontsize=14, weight="bold", pad=15, loc="left")
    axis.grid(axis="y", color="#dae0e6", linewidth=.7, zorder=0)
    axis.tick_params(axis="x", length=0, pad=12)
    axis.tick_params(axis="y", length=0, labelsize=10)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color("#b6c0cc")


def render(summary_path, output_prefix):
    summary = json.loads(summary_path.read_text())
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "pdf.fonttype": 42,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    figure, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1., 1.8]})
    groups = summary["groups"]
    draw_panel(axes[0], [("位置随机\n固定红色", "#b8434b", groups["position_red"]),
                         ("位置＋颜色随机", "#517ea9", groups["position_color"])], "两组评估")
    color_styles = [("red", "红", "#c74751"), ("green", "绿", "#65a174"),
                    ("blue", "蓝", "#5687ba"), ("white", "白", "#e1e5e9"),
                    ("yellow", "黄", "#e5bf51")]
    draw_panel(axes[1], [(name, color, groups["position_color"]["per_color"][key])
                         for key, name, color in color_styles], "位置＋颜色随机：按颜色统计")
    done = summary["complete"]
    status = "全部完成" if done else "评估进行中 · 阶段结果"
    figure.suptitle("PAP 抓取放盒 · 矩形区域随机评估", x=.065, y=.98, ha="left",
                     fontsize=20, weight="bold", color="#172b43")
    figure.text(.065, .925, f"{status}    已完成 {summary['completed']} / 计划 {summary['planned']} 个 episode",
                 fontsize=12, color="#4c5e71")
    if summary.get("sampling_region_matches_request") is not True:
        figure.text(.065, .88, "注意：此统计来自旧的错误采样范围，不能作为目标矩形区域的成功率。",
                     color="#b42332", fontsize=11)
    figure.text(.065, .075, "误差线：95% Wilson 置信区间。成功率分母为已完成的有效 episode；待评估和基础设施错误不计为失败。",
                 fontsize=10, color="#4c5e71")
    figure.text(.065, .04, "四个绿色标签沿长边延伸形成矩形边界；3 cm 方块完整位于矩形内。两组位置配对，EMA 平滑 α = 0.8。",
                 fontsize=10, color="#4c5e71")
    figure.subplots_adjust(left=.065, right=.98, top=.82, bottom=.27, wspace=.23)
    for extension in ("png", "pdf"):
        path = output_prefix.with_suffix("."+extension)
        figure.savefig(path, dpi=210, facecolor="white")
        print(path)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, help="Default: success_rates beside the summary")
    args = parser.parse_args()
    render(args.summary, args.output_prefix or args.summary.parent / "success_rates")


if __name__ == "__main__":
    main()
