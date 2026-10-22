#!/usr/bin/env python3
"""Offline spatial analysis of the corrected rectangle evaluation.

Run with --preview while episodes are still running. Without --preview this
requires all 100 fixed-red episodes to have completed. Outputs are written
under the selected evaluation root's location_analysis folder.
Dependencies: numpy, matplotlib, Pillow; a Chinese font is selected locally.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image


GROUP = "position_red"
REGION_KIND = "rectangle_from_marker_long_axis_lines"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def count(rows):
    failures = sum(not row["success"] for row in rows)
    return dict(total=len(rows), failures=failures, successes=len(rows)-failures,
                failure_rate=failures/len(rows) if rows else None,
                failure_episode_ids=[r["episode_id"] for r in rows if not r["success"]])


def load_records(root, plan, source):
    """Use only finished, valid episodes; running/missing episodes are pending."""
    planned = [ep for ep in plan["episodes"] if ep["group"] == GROUP]
    summary_rows = {}
    if source != "reports" and (root / "summary.json").exists():
        summary = read_json(root / "summary.json")
        if summary.get("region", {}).get("region_kind") != REGION_KIND:
            raise ValueError("summary.json does not describe the corrected rectangle")
        summary_rows = {r["episode_id"]: r for r in summary.get("episodes", [])}
    completed, pending, infrastructure = [], [], []
    for ep in planned:
        episode_id = ep["episode_id"]
        report_path = root / "run" / "episodes" / episode_id / "report.json"
        report = read_json(report_path) if source != "summary" and report_path.exists() else None
        row = summary_rows.get(episode_id)
        status = report.get("evaluation_status", report.get("status", "complete")) if report is not None else (row or {}).get("status")
        # Match pressb.pick_place.evaluation.normalize_result: interrupted or
        # malformed reports are infrastructure errors; invalid model actions
        # remain completed failures, even when they carry an error message.
        if report is not None:
            reason = report.get("failure_reason")
            error = report.get("evaluation_error", report.get("error"))
            invalid_report = (status not in ("complete", "completed")
                              or reason == "infrastructure_error"
                              or bool(error and reason != "invalid_policy_action")
                              or not isinstance(report.get("physical_success"), bool))
        else:
            invalid_report = status == "infrastructure_error"
        if invalid_report:
            infrastructure.append(episode_id)
            continue
        if report is not None:
            observed = np.asarray(report["evaluation_episode"]["center_xy"], dtype=float)
            if not np.allclose(observed, ep["center_xy"], rtol=0, atol=1e-8):
                raise ValueError(f"Plan/report position mismatch: {episode_id}")
            physical = report.get("physical_outcome", {})
            record = dict(episode_id=episode_id, success=report["physical_success"],
                          failure_reason=None if report["physical_success"] else report.get("failure_reason") or "timeout",
                          first_success_time_s=physical.get("first_success_time_s"),
                          source=str(report_path.relative_to(root)))
            if "initial_object_position_world" in report:
                actual = np.asarray(report["initial_object_position_world"][:2])
                base = np.asarray(report["scene"]["robot_base_position_world"][:2])
                record["actual_initial_xy_m"] = (actual-base).tolist()
                if not np.allclose(actual-base, ep["center_xy"], rtol=0, atol=5e-4):
                    raise ValueError(f"Actual/plan position differs by >0.5 mm: {episode_id}")
        elif row and row.get("completed") and isinstance(row.get("success"), bool):
            if not np.allclose(row["center_xy"], ep["center_xy"], rtol=0, atol=1e-8):
                raise ValueError(f"Plan/summary position mismatch: {episode_id}")
            record = dict(episode_id=episode_id, success=row["success"],
                          failure_reason=row.get("failure_reason"),
                          first_success_time_s=row.get("first_success_time_s"), source="summary.json")
        else:
            pending.append(episode_id)
            continue
        record.update(center_xy_m=ep["center_xy"], pair_index=ep.get("pair_index"))
        completed.append(record)
    return planned, completed, pending, infrastructure


def geometry(plan):
    region = plan["region"]
    if plan.get("version", 0) < 2 or region.get("region_kind") != REGION_KIND:
        raise ValueError("This analysis requires the corrected, marker-edge rectangle plan (version 2+)")
    outer = np.asarray(region["marker_polygon_xy"], dtype=float)
    valid = np.asarray(region["valid_center_polygon_xy"], dtype=float)
    lo, hi = valid.min(axis=0), valid.max(axis=0)
    if len(valid) != 4 or not np.all(np.isclose(valid, lo) | np.isclose(valid, hi)):
        raise ValueError("This plot uses robot X/Y bins and requires an axis-aligned rectangle")
    if not np.isclose(region["valid_center_area_m2"], np.prod(hi-lo)):
        raise ValueError("Valid-center rectangle area disagrees with its vertices")
    return outer, valid, lo, hi, np.linspace(lo[0], hi[0], 4), np.linspace(lo[1], hi[1], 4)


def background_and_camera(root, config, requested):
    path = root / requested if requested else next(iter(sorted((root / "run/episodes").glob("red_*/initial_up.jpg"))), None)
    if path is None or not path.exists():
        raise ValueError("No real initial_up.jpg exists in this evaluation; supply --background")
    image = Image.open(path).convert("RGB")
    report_path = path.parent / "report.json"
    report = read_json(report_path) if report_path.exists() else {}
    observed = report.get("cameras", {}).get("up", {})
    if "world_from_camera_usd_at_initial" in observed:
        transform = np.asarray(observed["world_from_camera_usd_at_initial"], dtype=float)
        transform[:3, 3] -= report["scene"]["robot_base_position_world"]
        intrinsics = np.asarray(observed["intrinsics"], dtype=float)
        provenance = str(report_path.relative_to(root)) + ":cameras.up"
    else:
        cam = config["up_camera"]
        transform = np.asarray(cam["base_from_camera_usd"], dtype=float)
        intrinsics = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1.]])
        provenance = "effective_config.json:up_camera"
    if image.size != tuple(config["up_camera"]["resolution"]):
        raise ValueError("Background resolution does not match the fixed-camera calibration")

    def project(points_xy, z=0.0):
        xy = np.asarray(points_xy, dtype=float).reshape(-1, 2)
        xyz = np.column_stack((xy, np.full(len(xy), z)))
        camera_usd = (xyz-transform[:3, 3]) @ transform[:3, :3]
        camera_cv = camera_usd * [1, -1, -1]
        if np.any(camera_cv[:, 2] <= 0):
            raise ValueError("A projected point is behind the fixed camera")
        homogeneous = camera_cv @ intrinsics.T
        return homogeneous[:, :2] / homogeneous[:, 2, None]

    reference = np.array(config["cube"]["center_xy"], dtype=float)
    a, b = project([reference, reference+[0, .01]])
    if b[0] <= a[0]:
        raise ValueError("Unexpected camera orientation: +Y must point right")
    return path, image, project, provenance


def measure(rows, config, lo, hi, x_edges, y_edges):
    for row in rows:
        x, y = row["center_xy_m"]
        if np.any(np.asarray([x, y]) < lo-1e-8) or np.any(np.asarray([x, y]) > hi+1e-8):
            raise ValueError(f"Completed point outside valid center region: {row['episode_id']}")
        row["x_bin"] = int(np.clip(np.searchsorted(x_edges, x, side="right")-1, 0, 2))
        row["y_bin"] = int(np.clip(np.searchsorted(y_edges, y, side="right")-1, 0, 2))
        row["robot_xy_distance_m"] = float(np.linalg.norm([x, y]))
        row["box_xy_distance_m"] = float(np.linalg.norm(np.array([x, y])-config["box"]["center_xy"]))
        row["center_boundary_distance_m"] = float(min(x-lo[0], hi[0]-x, y-lo[1], hi[1]-y))
    cells = []
    for i in range(3):
        for j in range(3):
            cells.append(dict(x_bin=i, y_bin=j, x_range_m=x_edges[i:i+2].tolist(),
                              y_range_m=y_edges[j:j+2].tolist(),
                              **count([r for r in rows if r["x_bin"] == i and r["y_bin"] == j])))
    assert sum(c["total"] for c in cells) == len(rows)
    assert sum(c["failures"] for c in cells) == count(rows)["failures"]
    partitions = {"y_negative": count([r for r in rows if r["center_xy_m"][1] < 0]),
                  "y_nonnegative": count([r for r in rows if r["center_xy_m"][1] >= 0])}
    for axis in ("x", "y"):
        for k in range(3):
            partitions[f"{axis}_third_{k+1}"] = count([r for r in rows if r[f"{axis}_bin"] == k])
    for threshold in (.01, .02):
        for close in (True, False):
            key = f"boundary_{'lt' if close else 'ge'}_{round(threshold*100)}cm"
            partitions[key] = count([r for r in rows if (r["center_boundary_distance_m"] < threshold) == close])
    distances = {}
    for outcome, success in (("success", True), ("failure", False)):
        selected = [r for r in rows if r["success"] == success]
        distances[outcome] = {"count": len(selected)}
        for key in ("robot_xy_distance_m", "box_xy_distance_m", "center_boundary_distance_m"):
            values = np.asarray([r[key] for r in selected])
            distances[outcome][key] = dict(mean=float(values.mean()), median=float(np.median(values)),
                                           min=float(values.min()), max=float(values.max())) if len(values) else None
    return cells, partitions, distances


def chinese_font():
    candidates = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                  "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"]
    for candidate in candidates:
        if Path(candidate).exists():
            font_manager.fontManager.addfont(candidate)
            return font_manager.FontProperties(fname=candidate).get_name()
    for family in ("Noto Sans CJK SC", "Microsoft YaHei", "SimHei"):
        if any(f.name == family for f in font_manager.fontManager.ttflist):
            return family
    raise RuntimeError("Install a Chinese font such as Noto Sans CJK before drawing")


def draw(output, summary, rows, image, project, outer, valid, x_edges, y_edges):
    plt.rcParams.update({"font.family": chinese_font(), "axes.unicode_minus": False,
                         "font.size": 12, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(16, 8.3), facecolor="white")
    scene = fig.add_axes([.025, .205, .585, .625])
    grid = fig.add_axes([.69, .245, .28, .53])
    overall = summary["overall"]
    prefix = "预览 · " if summary["preview"] else ""
    fig.text(.035, .95, "固定红色方块：初始位置与成败分布", fontsize=24, weight="bold", color="#222222")
    fig.text(.035, .895, f"{prefix}已完成 {overall['total']}/{summary['planned']} 次  ·  成功 {overall['successes']} 次  ·  失败 {overall['failures']} 次",
             fontsize=15, color="#A45B00" if summary["preview"] else "#444444")
    scene.imshow(image)
    scene.set_title("仿真固定第三视角：新矩形边界与初始点位", loc="left", pad=12, fontsize=15)
    for polygon, color, style, width in ((outer, "#F0E442", "-", 2.5), (valid, "#56B4E9", "--", 2.1)):
        scene.add_patch(Polygon(project(polygon), closed=True, fill=False, edgecolor=color,
                                linestyle=style, linewidth=width, zorder=3))
    for x in x_edges[1:-1]:
        line = project([[x, y_edges[0]], [x, y_edges[-1]]])
        scene.plot(line[:, 0], line[:, 1], color="white", alpha=.5, linewidth=.8, linestyle=":")
    for y in y_edges[1:-1]:
        line = project([[x_edges[0], y], [x_edges[-1], y]])
        scene.plot(line[:, 0], line[:, 1], color="white", alpha=.5, linewidth=.8, linestyle=":")
    for success in (True, False):
        selected = [r for r in rows if r["success"] == success]
        if not selected:
            continue
        points = project([r["center_xy_m"] for r in selected])
        if success:
            scene.scatter(*points.T, s=30, facecolors="#56B4E9", edgecolors="white", linewidths=.7, alpha=.88, zorder=4)
        else:
            scene.scatter(*points.T, s=76, marker="X", facecolors="#D55E00", edgecolors="white", linewidths=.7, zorder=5)
    scene.set_xlim(0, image.width)
    scene.set_ylim(image.height, 0)
    scene.axis("off")
    scene.text(.025, .055, "+Y → 画面右侧 / 盒子方向", transform=scene.transAxes,
               color="white", fontsize=11, bbox=dict(facecolor="#222222", alpha=.72, edgecolor="none", pad=5))
    handles = [Line2D([], [], color="#B0A521", lw=2.5, label="四条绿标签长轴构成的矩形"),
               Line2D([], [], color="#0072B2", ls="--", lw=2, label="方块中心可采样范围"),
               Line2D([], [], marker="o", color="none", markerfacecolor="#56B4E9", markeredgecolor="#444444", markersize=7, label="成功"),
               Line2D([], [], marker="X", color="none", markerfacecolor="#D55E00", markeredgecolor="white", markersize=9, label="失败")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.04, .125), ncol=2,
               frameon=False, fontsize=11, columnspacing=1.5, handlelength=2.2)
    cmap = LinearSegmentedColormap.from_list("failure_rate", ["#FFF8EF", "#E7AA68", "#A43F00"])
    cmap.set_bad("#ECECEC")
    rates = np.full((3, 3), np.nan)
    for cell in summary["grid_3x3"]:
        if cell["total"]:
            rates[cell["x_bin"], cell["y_bin"]] = cell["failure_rate"]
    mesh = grid.pcolormesh(y_edges*100, x_edges*100, np.ma.masked_invalid(rates),
                          cmap=cmap, norm=Normalize(0, 1), edgecolors="#FFFFFF", linewidth=3)
    for cell in summary["grid_3x3"]:
        i, j = cell["x_bin"], cell["y_bin"]
        cx, cy = np.mean(y_edges[j:j+2])*100, np.mean(x_edges[i:i+2])*100
        rate = cell["failure_rate"]
        text = f"{cell['failures']} / {cell['total']}\n{rate:.0%}" if cell["total"] else "0 / 0\n暂无记录"
        grid.text(cx, cy, text, ha="center", va="center", fontsize=17, linespacing=1.6,
                  color="white" if rate is not None and rate > .6 else "#222222", weight="bold")
    grid.set_title("中心范围三等分：失败数 / 总数", fontsize=15, loc="left", pad=18)
    grid.invert_yaxis()
    grid.set_xticks(y_edges*100, [f"{v*100:.1f}" for v in y_edges])
    grid.set_yticks(x_edges*100, [f"{v*100:.1f}" for v in x_edges])
    grid.set_xlabel("Y（厘米） → 画面右侧", labelpad=10)
    grid.set_ylabel("X（厘米） → 画面下方", labelpad=9)
    grid.tick_params(length=0)
    for spine in grid.spines.values():
        spine.set_visible(False)
    cax = fig.add_axes([.76, .135, .16, .017])
    bar = fig.colorbar(mesh, cax=cax, orientation="horizontal", ticks=[0, .5, 1])
    bar.ax.set_xticklabels(["0%", "50%", "100%"])
    bar.ax.tick_params(labelsize=10, length=0)
    bar.outline.set_visible(False)
    fig.text(.69, .143, "失败率", fontsize=11)
    unfinished = "未完成试验不计入分母；当前图仅为预览。" if summary["preview"] else "每格分母均为该格实际完成试验数。"
    fig.text(.035, .075, f"{unfinished}按新中心范围的 X、Y 各三等分，未按成败调整边界。", fontsize=11, color="#555555")
    fig.text(.035, .038, f"背景：{summary['background']}；散点表示初始中心的桌面投影。位置关联不等于失败原因。", fontsize=10, color="#666666")
    fig.savefig(output.with_suffix(".png"), dpi=180, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def write_csv(path, rows):
    fields = ["episode_id", "success", "x_m", "y_m", "x_cm", "y_cm", "x_third", "y_third",
              "failure_reason", "robot_xy_distance_cm", "box_xy_distance_cm", "center_boundary_distance_cm"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            x, y = row["center_xy_m"]
            writer.writerow(dict(episode_id=row["episode_id"], success=row["success"], x_m=x, y_m=y,
                                 x_cm=x*100, y_cm=y*100, x_third=row["x_bin"]+1, y_third=row["y_bin"]+1,
                                 failure_reason=row["failure_reason"],
                                 robot_xy_distance_cm=row["robot_xy_distance_m"]*100,
                                 box_xy_distance_cm=row["box_xy_distance_m"]*100,
                                 center_boundary_distance_cm=row["center_boundary_distance_m"]*100))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Evaluation directory containing plan.json and run/")
    parser.add_argument("--preview", action="store_true", help="Allow partial data; name and label every output as preview")
    parser.add_argument("--source", choices=["auto", "reports", "summary"], default="auto",
                        help="auto: prefer completed episode reports, with summary.json as fallback")
    parser.add_argument("--background", help="Path to a recorded simulation initial_up.jpg relative to the evaluation root")
    args = parser.parse_args()
    root = args.root.resolve()
    plan = read_json(root / "plan.json")
    config = read_json(root / "effective_config.json")
    outer, valid, lo, hi, x_edges, y_edges = geometry(plan)
    planned, rows, pending, infrastructure = load_records(root, plan, args.source)
    if len(planned) != 100:
        parser.error(f"Expected 100 fixed-red planned episodes; found {len(planned)}")
    if not args.preview and (len(rows) != 100 or infrastructure):
        parser.error(f"Fixed-red run is not complete: {len(rows)}/100 valid, {len(infrastructure)} infrastructure errors. Use --preview for partial output.")
    cells, partitions, distances = measure(rows, config, lo, hi, x_edges, y_edges)
    background, image, project, camera_source = background_and_camera(root, config, args.background)
    summary = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(), group=GROUP,
                   preview=args.preview, complete=len(rows) == len(planned) and not infrastructure,
                   planned=len(planned), overall=count(rows), pending_episode_ids=pending,
                   infrastructure_error_episode_ids=infrastructure, region_kind=REGION_KIND,
                   marker_polygon_xy_m=outer.tolist(), valid_center_polygon_xy_m=valid.tolist(),
                   x_grid_edges_m=x_edges.tolist(), y_grid_edges_m=y_edges.tolist(),
                   grid_3x3=cells, partitions=partitions, distances=distances,
                   source_counts=dict(Counter(r["source"].split(":")[0] for r in rows)),
                   background=str(background.relative_to(root)), camera_source=camera_source,
                   episodes=rows, interpretation="Descriptive location association only, not failure mechanism or causation.",
                   visual_contract=dict(format="Chinese local analysis figure, PNG and PDF",
                       question="Where did failures occur relative to the actual sampled positions?",
                       evidence="Completed fixed-red episodes from the new rectangle run only",
                       panels=["Recorded simulation fixed-camera frame with true rectangle and tabletop-projected outcome points",
                               "3x3 equal-width valid-center bins, each labeled failures / completed attempts"],
                       statistics="No smoothing, optimized thresholds, significance claims, or old-run values",
                       grid_convention="Rows increase in X downward; columns increase in Y rightward; final bin includes upper edge",
                       colors="Success blue circle; failure vermilion X; fixed 0–100% sequential failure-rate scale"))
    output = root / "location_analysis"
    output.mkdir(parents=True, exist_ok=True)
    prefix = "preview_" if args.preview else ""
    draw(output / f"{prefix}failure_locations", summary, rows, image, project, outer, valid, x_edges, y_edges)
    (output / f"{prefix}summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    write_csv(output / f"{prefix}failed_initial_positions.csv", [r for r in rows if not r["success"]])
    write_csv(output / f"{prefix}all_initial_positions.csv", rows)
    print(json.dumps(dict(preview=args.preview, completed=len(rows), planned=len(planned),
                          failures=summary["overall"]["failures"], output=str(output)), ensure_ascii=False))


if __name__ == "__main__":
    main()
