"""Fixed evaluation layouts and schedules, without expert motion planning.

These coordinates configure the simulated scene only. Neither the policy
request nor the pose controller receives layout or target-button coordinates.
"""
from __future__ import annotations

import math
from numbers import Integral, Real


LAYOUT_FIELDS = ("panel_layout_index", "panel_layout_name", "panel_offset_x_m", "panel_offset_y_m")


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def evaluation_panel_layouts(cfg, mode="fixed"):
    """Return fixed positions from the frozen training scene configuration."""
    if mode == "fixed":
        positions = [("fixed", _finite(cfg.get("panel_offset_x_m", 0.), "panel_offset_x_m"),
                      _finite(cfg.get("panel_offset_y_m", 0.), "panel_offset_y_m"))]
    elif mode == "center_corners":
        settings = cfg.get("panel_randomization", {})
        if not isinstance(settings, dict) or settings.get("enabled") is not True:
            raise ValueError("center_corners requires the recorded panel_randomization bounds to be enabled")
        bounds = {}
        for axis in "xy":
            low, high = (_finite(settings.get(f"{which}_offset_{axis}_m"), f"{which}_offset_{axis}_m")
                         for which in ("min", "max"))
            if not low < high:
                raise ValueError(f"Evaluation panel {axis} bounds must have positive width")
            bounds[axis] = (low, high)
        xmin, xmax = bounds["x"]
        ymin, ymax = bounds["y"]
        positions = [("center", xmin / 2 + xmax / 2, ymin / 2 + ymax / 2),
                     ("xmin_ymin", xmin, ymin), ("xmin_ymax", xmin, ymax),
                     ("xmax_ymin", xmax, ymin), ("xmax_ymax", xmax, ymax)]
    else:
        raise ValueError("panel layout mode must be fixed or center_corners")
    return [dict(panel_layout_index=i, panel_layout_name=name, panel_offset_x_m=x, panel_offset_y_m=y)
            for i, (name, x, y) in enumerate(positions)]


def evaluation_schedule(floors, episodes_per_floor, layouts):
    """Each repeat visits every layout, then each requested floor.

    The canonical 12-floor seed index stays stable when selecting fewer floors
    or changing the number of simultaneous environments. One fixed layout
    reproduces the previous evaluation schedule and inference seeds exactly.
    """
    floors = list(floors)
    if (not floors or any(isinstance(f, bool) or not isinstance(f, Integral) or not 24 <= f <= 35 for f in floors)
            or len(floors) != len(set(floors))):
        raise ValueError("Evaluation floors must be distinct integers from 24 through 35")
    if isinstance(episodes_per_floor, bool) or not isinstance(episodes_per_floor, Integral) or episodes_per_floor < 1:
        raise ValueError("episodes_per_floor must be a positive integer")
    if not layouts or [p.get("panel_layout_index") for p in layouts] != list(range(len(layouts))):
        raise ValueError("Evaluation layouts require consecutive panel_layout_index values")
    schedule = []
    for repeat in range(episodes_per_floor):
        for layout in layouts:
            for floor in floors:
                schedule.append(dict(episode_id=len(schedule), floor=int(floor), repeat=repeat,
                                     seed_episode_index=(repeat * len(layouts) + layout["panel_layout_index"]) * 12 + floor - 24,
                                     **{key: layout[key] for key in LAYOUT_FIELDS}))
    return schedule


def reposition_evaluation_panels(world, envs, batch_schedule):
    """Apply absolute XY offsets after arms reset; the caller then settles physics.

    Kept separate from stepping/capture so all environments can finish their
    scene edits before the shared World advances. No IK or trajectory is used.
    """
    from .dataset_scene import set_panel_offset

    if len(batch_schedule) > len(envs):
        raise ValueError("More evaluation layouts than environments")
    for env, episode in zip(envs, batch_schedule):
        set_panel_offset(world, env, offset_y_m=episode["panel_offset_y_m"],
                         offset_x_m=episode["panel_offset_x_m"])


def evaluation_layout_evidence(world, env, episode):
    """Read settled USD/PhysX evidence and reject a stale layout before imaging."""
    from .dataset_scene import validate_panel_layout

    evidence = validate_panel_layout(world, env)
    for axis in "xy":
        field = f"panel_offset_{axis}_m"
        if evidence[field] != episode[field]:
            raise RuntimeError(f"Settled evaluation layout has the wrong {axis} offset")
    return evidence
