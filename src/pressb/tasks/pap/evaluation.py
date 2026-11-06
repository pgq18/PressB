"""Reproducible placement plans and honest partial PAP evaluation summaries.

This module has no Isaac Sim or model dependencies. Coordinates are metres in
the robot base frame; sampling is uniform over the entire valid centre polygon.
"""
from __future__ import annotations

import copy
import csv
import json
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull


COLORS = {
    "red": [.7, .025, .035],
    "green": [.025, .55, .045],
    "blue": [.006, .36, .85],
    "white": [.75, .75, .75],
    "yellow": [.8, .64, .015],
}
GROUPS = ("position_red", "position_color")
REGION_KIND = "rectangle_from_marker_long_axis_lines"
LEGACY_REGION_NOTICE = (
    "Incorrect sampling region for the user's task: this legacy plan treated the four tape "
    "centres as corner vertices. The markers instead identify four rectangle edges. "
    "These episode counts describe only the old inner quadrilateral and must not be reported "
    "as success rates over the requested rectangle. Existing outcomes are retained unchanged."
)


def placement_region(config, edge=.03, yaw_deg=0.):
    """Intersect four marker long-axis lines, then erode by the cube footprint.

    Each tape centre lies on one rectangle edge; it is not a polygon corner.
    Tape width does not shift the boundary: the line through its centre along
    its long axis is the explicit geometric convention for the placement area.
    """
    markers = config["tape_markers"]
    points = np.asarray([marker["xy"] for marker in markers], dtype=float)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("Expected exactly four finite tabletop marker coordinates")
    try:
        yaws = np.asarray([marker["yaw_deg"] for marker in markers], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Each rectangle-edge marker requires a finite long-axis yaw_deg") from error
    if yaws.shape != (4,) or not np.isfinite(yaws).all():
        raise ValueError("Each rectangle-edge marker requires a finite long-axis yaw_deg")
    if not np.isfinite([edge, yaw_deg]).all() or edge <= 0:
        raise ValueError("Cube edge must be positive and yaw finite")
    marker_angles = np.deg2rad(yaws)
    directions = np.column_stack((np.cos(marker_angles), np.sin(marker_angles)))
    first_axis = directions[0]
    second_axis = np.array([-first_axis[1], first_axis[0]])
    alignment = np.abs(directions @ first_axis)
    parallel = np.isclose(alignment, 1., rtol=0., atol=1e-10)
    perpendicular = np.isclose(alignment, 0., rtol=0., atol=1e-10)
    if parallel.sum() != 2 or perpendicular.sum() != 2:
        raise ValueError("Marker long axes must form two parallel pairs of perpendicular rectangle edges")
    basis = np.column_stack((first_axis, second_axis))
    coordinates = points @ basis
    lower = np.array([coordinates[perpendicular, 0].min(), coordinates[parallel, 1].min()])
    upper = np.array([coordinates[perpendicular, 0].max(), coordinates[parallel, 1].max()])
    if np.any(upper-lower <= 1e-10):
        raise ValueError("Opposite rectangle marker edges must be distinct")
    for i, local in enumerate(coordinates):
        along = 0 if parallel[i] else 1
        if not lower[along]+1e-10 < local[along] < upper[along]-1e-10:
            raise ValueError("Each tape centre must lie inside its rectangle edge segment, not beyond a corner")
    corners = np.array([[lower[0], lower[1]], [upper[0], lower[1]],
                        [upper[0], upper[1]], [lower[0], upper[1]]]) @ basis.T
    hull = ConvexHull(corners)
    normals, offsets = hull.equations[:, :2], hull.equations[:, 2]
    edge_lines = []
    for i, point in enumerate(points):
        boundary = int(np.argmin(np.abs(normals @ point + offsets)))
        edge_lines.append(dict(marker_index=i, point_xy=point.tolist(), yaw_deg=float(yaws[i]),
                               long_axis_xy=directions[i].tolist(),
                               outward_normal_xy=normals[boundary].tolist(),
                               offset=float(offsets[boundary])))
    angle = np.deg2rad(yaw_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    support = edge/2 * np.abs(normals @ rotation).sum(axis=1)
    eroded_offsets = offsets + support
    candidates = []
    for i in range(len(normals)):
        for j in range(i):
            matrix = normals[[j, i]]
            if abs(np.linalg.det(matrix)) < 1e-12:
                continue
            point = np.linalg.solve(matrix, -eroded_offsets[[j, i]])
            if np.all(normals @ point + eroded_offsets <= 1e-10):
                if not any(np.linalg.norm(point-other) < 1e-10 for other in candidates):
                    candidates.append(point)
    if len(candidates) < 3:
        raise ValueError("Cube footprint leaves no positive-area placement region")
    candidates = np.asarray(candidates)
    inner_hull = ConvexHull(candidates)
    inner = candidates[inner_hull.vertices]
    return dict(region_kind=REGION_KIND, marker_centres_xy=points.tolist(),
                marker_polygon_xy=corners[hull.vertices].tolist(), edge_lines=edge_lines,
                tape_metadata=dict(long_axis_size_m=.035, short_axis_size_m=.012,
                                   boundary_convention="Line through tape centre along its long axis; tape width is not subtracted"),
                marker_polygon_area_m2=float(hull.volume), valid_center_polygon_xy=inner.tolist(),
                valid_center_area_m2=float(inner_hull.volume),
                valid_center_bounds_xy_m=[inner.min(axis=0).tolist(), inner.max(axis=0).tolist()],
                marker_halfplanes=np.column_stack((normals, offsets)).tolist(),
                valid_center_halfplanes=np.column_stack((normals, eroded_offsets)).tolist(),
                edge_m=float(edge), yaw_deg=float(yaw_deg),
                interpretation="Rectangle bounded by the four green tape long-axis centre lines; entire cube footprint stays inside")


def sample_polygon(polygon, count, rng):
    """Area-weighted triangle sampling, without workspace/reachability rejection."""
    polygon = np.asarray(polygon, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("Expected an ordered convex polygon")
    triangles = np.array([[polygon[0], polygon[i], polygon[i+1]] for i in range(1, len(polygon)-1)])
    areas = np.abs(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]))/2
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError("Placement polygon must have positive area")
    selected = triangles[rng.choice(len(triangles), size=count, p=areas/areas.sum())]
    root_u, v = np.sqrt(rng.random(count)), rng.random(count)
    return ((1-root_u)[:, None]*selected[:, 0]
            + (root_u*(1-v))[:, None]*selected[:, 1] + (root_u*v)[:, None]*selected[:, 2])


def _validate_evaluation_preset(config):
    """Keep the fixed experiment's declared geometry and timing true at runtime."""
    object_key = "object" if "object" in config else "cube"
    obj = config.get(object_key)
    if (not isinstance(obj, Mapping)
            or object_key == "object" and obj.get("shape") != "cube"):
        raise ValueError("PAP evaluation preset requires a 3 cm cube")
    from .task import object_geometry
    geometry = object_geometry({**obj, "shape": "cube"})
    expected = {
        "cube edge": (geometry["edge"], .03),
        "cube grasp width": (geometry["grasp_width_m"], .03),
        "cube mass": (obj.get("mass", .035), .025),
        "cube yaw": (obj.get("yaw_deg", 0.), 0.),
        "physics_dt": (config.get("physics_dt"), 1/120),
        "fps": (config.get("fps"), 30),
        "gripper_max_width_m": (config.get("gripper_max_width_m"), .1),
    }
    for name, (actual, wanted) in expected.items():
        try:
            matches = math.isclose(float(actual), wanted, rel_tol=0., abs_tol=1e-12)
        except (TypeError, ValueError):
            matches = False
        if not matches:
            raise ValueError(f"PAP evaluation preset requires {name}={wanted}, got {actual!r}")
    source = config.get("episode_source", {})
    if (not isinstance(source, Mapping) or source.get("episode_index", 0) != 0
            or source.get("initial_frame", 0) != 0):
        raise ValueError("PAP evaluation preset requires initial dataset episode 0, frame 0")
    trajectory = source.get("trajectory")
    if trajectory is not None and Path(trajectory).stem != "episode_000000":
        raise ValueError("PAP evaluation preset requires the episode_000000 initialization trajectory")
    if not config.get("task_props", True):
        raise ValueError("PAP evaluation preset requires the tabletop object and box")


def make_plan(config, seed=20261023, episodes_per_group=100, duration_s=20.):
    if isinstance(episodes_per_group, bool) or episodes_per_group <= 0 or episodes_per_group % 5:
        raise ValueError("episodes_per_group must be a positive multiple of five")
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    _validate_evaluation_preset(config)
    region = placement_region(config)
    position_seed, color_seed = np.random.SeedSequence(seed).spawn(2)
    positions = sample_polygon(region["valid_center_polygon_xy"], episodes_per_group,
                               np.random.default_rng(position_seed))
    colors = np.tile(list(COLORS), episodes_per_group//5)
    np.random.default_rng(color_seed).shuffle(colors)
    episodes = []
    recorded_colors = set()
    for group in GROUPS:
        for i, position in enumerate(positions):
            color = "red" if group == "position_red" else str(colors[i])
            record_video = i < 2 if group == "position_red" else color not in recorded_colors
            if group == "position_color":
                recorded_colors.add(color)
            episodes.append(dict(episode_id=f"{'red' if group == 'position_red' else 'color'}_{i:03d}",
                                 group=group, pair_index=i, color=color, center_xy=position.tolist(),
                                 color_rgb=COLORS[color].copy(), record_video=record_video))
    return dict(version=2, seed=int(seed), episodes_per_group=int(episodes_per_group),
                duration_s=float(duration_s), stop_on_success=True, episodes=episodes, region=region,
                protocol=dict(position_distribution="uniform area in valid_center_polygon_xy",
                              placement_boundary="Four rectangle edges are marker long-axis lines through their centres, not a hull of marker centres",
                              paired_positions=True, reachability_filter=False,
                              color_distribution="balanced random permutation; 20 per color for 100 episodes",
                              colors_linear_rgb=copy.deepcopy(COLORS),
                              material_note="Approximate linear RGB colors; only diffuse color changes",
                              cube_edge_m=.03, cube_mass_kg=.025, cube_yaw_deg=0.,
                              initial_dataset_episode=0, control_hz=30, physics_hz=120,
                              execute_steps=32, smoothing=dict(mode="joint_ema", alpha=.8),
                              model_path="/home/pengguanqi/Models/piper_pap_20260606",
                              normalization_path="/home/pengguanqi/Models/piper_pap_20260606/norm_stats.json",
                              arm_limit_mode="error", gripper_limit_mode="clip", gripper_width_limits_m=[0., .1],
                              success="Existing physical PAP outcome: lift, full containment, release, floor rest, stable 0.5 s, bilateral contact history",
                              timeout_rule=f"No physical success by {duration_s:g} simulated seconds is failure",
                              invalid_action_rule="Invalid model actions count as failures",
                              infrastructure_error_rule="Infrastructure errors invalidate completion; report separately without claiming 100 episodes",
                              diffusion_noise="Native model sampling; no per-request generator seed override"))


def effective_config(config):
    result = copy.deepcopy(config)
    result.setdefault("policy_execution", {})["smoothing"] = dict(mode="joint_ema", alpha=.8)
    return result


def wilson_interval(successes, total, z=1.959963984540054):
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("Invalid binomial counts")
    if total == 0:
        return None
    p, z2 = successes/total, z*z
    centre = (p+z2/(2*total))/(1+z2/total)
    radius = z*math.sqrt(p*(1-p)/total+z2/(4*total*total))/(1+z2/total)
    return [max(0., centre-radius), min(1., centre+radius)]


def _first_success(report):
    for record in (report, report.get("physical_outcome", {}), report.get("result", {}), report.get("outcome", {})):
        value = record.get("first_success_time_s")
        if value is not None:
            return float(value)
    return None


def normalize_result(episode, report):
    """A result is completed only with an explicit outcome and no infrastructure error."""
    row = {key: copy.deepcopy(episode[key]) for key in ("episode_id", "group", "color", "pair_index", "center_xy")}
    if report is None:
        return dict(row, status="pending", completed=False, success=None,
                    failure_reason=None, first_success_time_s=None, error=None)
    status = report.get("evaluation_status", report.get("status", "complete"))
    reason = report.get("failure_reason")
    error = report.get("evaluation_error", report.get("error"))
    infrastructure = (status not in ("complete", "completed")
                      or reason == "infrastructure_error"
                      or error and reason != "invalid_policy_action")
    physical_success = report.get("physical_success", report.get("success"))
    if infrastructure or not isinstance(physical_success, bool):
        return dict(row, status="infrastructure_error", completed=False, success=None,
                    failure_reason=reason or "infrastructure_error", first_success_time_s=None,
                    error=error or "Report has no explicit physical_success boolean")
    success = physical_success
    return dict(row, status="complete", completed=True, success=success,
                failure_reason=None if success else reason or "timeout", first_success_time_s=_first_success(report),
                error=error, simulated_duration_s=report.get("simulated_duration_s", report.get("duration_s")))


def group_statistics(rows, planned):
    complete = [row for row in rows if row["completed"]]
    successes = sum(row["success"] for row in complete)
    times = [row["first_success_time_s"] for row in complete
             if row["success"] and row["first_success_time_s"] is not None]
    return dict(planned=planned, completed=len(complete), successes=successes,
                failures=len(complete)-successes, pending=sum(row["status"] == "pending" for row in rows),
                infrastructure_errors=sum(row["status"] == "infrastructure_error" for row in rows),
                success_rate=successes/len(complete) if complete else None,
                wilson_95=wilson_interval(successes, len(complete)),
                mean_first_success_time_s=float(np.mean(times)) if times else None,
                failure_reasons=dict(Counter(row["failure_reason"] for row in complete if not row["success"])))


def summarize_results(plan, reports):
    rows = [normalize_result(episode, reports.get(episode["episode_id"])) for episode in plan["episodes"]]
    groups = {}
    for group in GROUPS:
        members = [row for row in rows if row["group"] == group]
        groups[group] = group_statistics(members, len(members))
        groups[group]["per_color"] = {
            color: group_statistics([row for row in members if row["color"] == color],
                                    sum(row["color"] == color for row in members))
            for color in COLORS if any(row["color"] == color for row in members)}
    correct_region = plan["region"].get("region_kind") == REGION_KIND
    return dict(version=2, seed=plan["seed"], complete=all(row["completed"] for row in rows),
                sampling_region_matches_request=correct_region,
                sampling_region_notice=None if correct_region else LEGACY_REGION_NOTICE,
                planned=len(rows), completed=sum(row["completed"] for row in rows), groups=groups,
                failures=[row for row in rows if row["completed"] and not row["success"]],
                infrastructure_errors=[row for row in rows if row["status"] == "infrastructure_error"],
                protocol=plan["protocol"], region=plan["region"], episodes=rows,
                interpretation="Rates use completed episodes including timeouts and invalid model actions; pending/infrastructure errors are shown separately. Completion requires all planned episodes.")


def read_reports(root):
    root = Path(root)
    reports = {}
    for path in (root/"run"/"results.jsonl", root/"results.jsonl"):
        if path.exists():
            lines = path.read_text().splitlines()
            for index, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    if index == len(lines)-1:
                        continue  # A live writer may not have finished its final row.
                    raise
                episode_id = record.get("episode_id", record.get("episode", {}).get("episode_id"))
                if episode_id is None:
                    raise ValueError(f"Result row in {path} has no episode_id")
                reports[episode_id] = record
    for path in sorted((root/"run"/"episodes").glob("*/report.json")):
        try:
            report = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue  # Do not interpret a partially written live report as a trial failure.
        reports[path.parent.name] = report
    return reports


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def write_summary(root):
    root = Path(root)
    plan = json.loads((root/"plan.json").read_text())
    summary = summarize_results(plan, read_reports(root))
    write_json(root/"summary.json", summary)
    with (root/"episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["episode_id", "group", "color", "pair_index", "x_m", "y_m",
                                                  "status", "success", "failure_reason", "first_success_time_s", "error"])
        writer.writeheader()
        for row in summary["episodes"]:
            writer.writerow({key: row.get(key) for key in writer.fieldnames if key not in ("x_m", "y_m")} |
                            dict(x_m=row["center_xy"][0], y_m=row["center_xy"][1]))
    lines = ["# PAP randomized evaluation", "", f"Completed: {summary['completed']}/{summary['planned']}. "
             + ("All planned trials completed." if summary["complete"] else "Partial result; evaluation is not complete."), "",
             "| Group / color | Success / completed | Planned | Rate | Wilson 95% CI |", "|---|---:|---:|---:|---:|"]
    if summary["sampling_region_notice"]:
        lines[2:2] = ["> **Sampling region correction:** " + summary["sampling_region_notice"], ""]
    for group, stats in summary["groups"].items():
        for label, values in [(group, stats)]+[(f"{group} / {c}", s) for c, s in stats["per_color"].items()]:
            rate = "—" if values["success_rate"] is None else f"{values['success_rate']:.1%}"
            ci = "—" if values["wilson_95"] is None else "–".join(f"{v:.1%}" for v in values["wilson_95"])
            lines.append(f"| {label} | {values['successes']}/{values['completed']} | {values['planned']} | {rate} | {ci} |")
    lines += ["", summary["interpretation"], "", "Physical simulation timeout: 20 s. Joint EMA: alpha = 0.8. "
              "Cube: 3 cm; yaw: 0 deg. Group positions are paired; color group is balanced across five colors.", ""]
    (root/"summary.md").write_text("\n".join(lines))
    return summary


def plot_plan(plan, path, summary=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    figure, axes = plt.subplots(1, 2 if summary else 1, figsize=(12 if summary else 7, 6), squeeze=False)
    correct_region = plan["region"].get("region_kind") == REGION_KIND
    for axis, group in zip(axes[0], GROUPS):
        axis.add_patch(Polygon(plan["region"]["marker_polygon_xy"], fill=False, edgecolor="#478748", linewidth=2,
                               label="Rectangle edge boundary" if correct_region else "Incorrect legacy marker-centre hull"))
        for i, line in enumerate(plan["region"].get("edge_lines", [])):
            centre = np.asarray(line["point_xy"])
            direction = np.asarray(line["long_axis_xy"])
            tape = centre + np.array([[-.0175], [.0175]])*direction
            axis.plot(tape[:, 0], tape[:, 1], color="#62a82c", linewidth=6,
                      label="Tape markers (edge segments)" if i == 0 else None)
        axis.add_patch(Polygon(plan["region"]["valid_center_polygon_xy"], color="#d8e8d7", alpha=.5,
                               label="Valid cube centres"))
        if summary:
            members = [row for row in summary["episodes"] if row["group"] == group]
            for state, color, marker in [(True, "#267b35", "o"), (False, "#ce3939", "x"), (None, "#999999", ".")]:
                xy = np.asarray([row["center_xy"] for row in members if row["success"] is state])
                if len(xy):
                    axis.scatter(xy[:, 0], xy[:, 1], c=color, marker=marker, s=32,
                                 label={True:"Success", False:"Failure", None:"Pending / error"}[state])
            stats = summary["groups"][group]
            axis.set_title(f"{group}: {stats['successes']}/{stats['completed']} successful\n"
                           f"({stats['completed']}/{stats['planned']} episodes completed)")
        else:
            xy = np.array([episode["center_xy"] for episode in plan["episodes"] if episode["group"] == group])
            axis.scatter(xy[:, 0], xy[:, 1], c="#bb2632", s=20, label=f"{len(xy)} paired positions")
            axis.set_title("Randomized PAP placement region")
        if not correct_region:
            axis.set_title("WRONG DOMAIN: legacy inner quadrilateral\n" + axis.get_title(), color="#a42626")
        axis.set(xlabel="Base X (m)", ylabel="Base Y (m)", aspect="equal")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
