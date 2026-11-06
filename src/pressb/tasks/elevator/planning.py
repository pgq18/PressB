"""Bounded joint trajectories solved from the official Piper URDF."""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import CubicSpline


def base_position(cfg):
    """World-frame mounting position shared by planning, rendering and audits."""
    return np.array([cfg.get("robot_base_x", 0.), cfg.get("robot_base_y", 0.), cfg["table_height"]])


def button_centers(cfg):
    face_x = panel_face_x(cfg)
    offset_y = float(cfg.get("panel_offset_y_m", 0.))
    if not np.isfinite(offset_y):
        raise ValueError("panel_offset_y_m must be finite")
    return {
        floor: np.array([face_x, offset_y + (1 if floor < 30 else -1) * cfg["button_column_y"],
                         cfg["button_bottom_z"] + (floor - (24 if floor < 30 else 30)) * cfg["button_pitch_z"]])
        for floor in range(24, 36)
    }


def panel_face_x(cfg):
    """Actual front face after the wall/panel assembly's fore-aft displacement."""
    face_x = float(cfg["button_face_x"]) + float(cfg.get("panel_offset_x_m", 0.))
    if not np.isfinite(face_x):
        raise ValueError("button_face_x and panel_offset_x_m must be finite")
    return face_x


@dataclass
class Trajectory:
    time: np.ndarray
    q: np.ndarray
    floor: np.ndarray
    phase: np.ndarray
    target_tip: np.ndarray


def make_plan(kin, cfg):
    # Tool +Z points toward the wall (+X); local +X points down (-Z).
    orientation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    base = base_position(cfg)
    centers = button_centers(cfg)
    face_x = panel_face_x(cfg)
    qs, floors, phases, tips = [], [], [], []
    first_floor = cfg["sequence"][0]
    home = np.asarray(cfg["home_q"], dtype=float)
    # A small shoulder adjustment to the official folded zero pose levels the
    # two dogleg housings as closely as their elbow limit allows (about 1 degree).
    # Direct unfolding is clear for this scene; a custom setup can request an
    # intermediate joint waypoint without changing the cycle-home semantics.
    clearance = np.asarray(cfg["home_clearance_q"], dtype=float) if "home_clearance_q" in cfg else None
    poses_to_check = [("home_q", home)]
    if clearance is not None:
        poses_to_check.append(("home_clearance_q", clearance))
    for name, pose in poses_to_check:
        if pose.shape != (6,) or not np.isfinite(pose).all() or np.any(pose < kin.lower) or np.any(pose > kin.upper):
            raise ValueError(f"{name} must contain six finite joint angles inside the official limits")
    current = home.copy()
    current_point = kin.fk(current)[:3, 3]

    def joint_segment(goal, duration, floor, phase):
        """Move smoothly between collision-checked configurations."""
        nonlocal current, current_point
        duration = max(duration, 1.05 * 1.875 * float(np.max(np.abs(goal - current))) / cfg["joint_speed"])
        count = max(2, int(np.ceil(duration / cfg["physics_dt"])))
        u = np.arange(1, count + 1) / count
        blend = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
        for weight in blend:
            q = current + weight * (goal - current)
            pose = kin.fk(q)
            tip = pose[:3, 3] + base
            # A spherical tip can project farther along world X than its axial tip.
            sphere_front_x = tip[0] + .005 * (1. - pose[0, 2])
            if sphere_front_x >= face_x - .005:
                raise RuntimeError(f"Home transfer enters the button clearance at floor {floor}")
            qs.append(q)
            floors.append(floor)
            phases.append(phase)
            tips.append(tip)
        current = goal.copy()
        current_point = kin.fk(current)[:3, 3]

    def segment(goal_point, duration, floor, phase):
        """Track a straight Cartesian path with smooth timing and bounded speed."""
        nonlocal current, current_point
        goal_point = np.asarray(goal_point, dtype=float)
        distance = float(np.linalg.norm(goal_point - current_point))
        if distance < 1e-10:
            count = max(2, int(np.ceil(duration / cfg["physics_dt"])))
            qs.extend([current.copy() for _ in range(count)])
            floors.extend([floor] * count)
            phases.extend([phase] * count)
            tips.extend([current_point + base for _ in range(count)])
            return

        # Solve in Cartesian order, so the wrist stays on a continuous IK branch.
        parameters = [0.]
        knots = [current.copy()]

        def add_knot(parameter, depth=0):
            point = current_point + parameter * (goal_point - current_point)
            q = kin.solve(point, knots[-1], rotation=orientation)
            if np.max(np.abs(q - knots[-1])) > 0.10:
                if depth >= 12:
                    raise RuntimeError(f"Discontinuous IK branch at floor {floor}, phase {phase}")
                add_knot((parameters[-1] + parameter) / 2, depth + 1)
                add_knot(parameter, depth + 1)
                return
            parameters.append(parameter)
            knots.append(q)

        for parameter in np.linspace(0., 1., max(3, int(np.ceil(distance / 0.004)) + 1))[1:]:
            add_knot(float(parameter))
        curve = CubicSpline(parameters, np.asarray(knots), axis=0, bc_type="natural")
        # The quintic time law has max ds/du=1.875. Include a sampling margin.
        max_derivative = float(np.max(np.abs(curve(np.linspace(0., 1., 256), 1))))
        duration = max(duration, 1.05 * 1.875 * max_derivative / cfg["joint_speed"])
        count = max(2, int(np.ceil(duration / cfg["physics_dt"])))
        u = np.arange(1, count + 1) / count
        blend = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
        for parameter in blend:
            q = curve(parameter)
            qs.append(q)
            floors.append(floor)
            phases.append(phase)
            tips.append(current_point + parameter * (goal_point - current_point) + base)
        current = knots[-1].copy()
        current_point = goal_point.copy()

    segment(current_point, 0.5, first_floor, "settle")
    for floor in cfg["sequence"]:
        front = centers[floor] - base
        approach = front - [cfg["approach_distance"], 0., 0.]
        pressed = front + [cfg["press_depth"], 0., 0.]
        # Every cycle starts at the same near-horizontal folded pose. The stock
        # castings have dogleg offsets, so joint-center vectors need not be level
        # for their long casing sections to appear horizontally stacked.
        qa = kin.solve(approach, [0., 1.5, -1.5, 0., -.5, 0.], rotation=orientation)
        if clearance is not None:
            joint_segment(clearance, 0.8, floor, "depart_home")
        joint_segment(qa, 0.8, floor, "approach")
        segment(pressed, cfg["press_duration"], floor, "press")
        segment(pressed, cfg["dwell_duration"], floor, "hold")
        segment(approach, cfg["retract_duration"], floor, "retract")
        if clearance is not None:
            joint_segment(clearance, 0.8, floor, "return_home")
        joint_segment(home, 0.8, floor, "return_home")
        segment(current_point, cfg["home_hold_duration"], floor, "home_hold")
    q_array = np.asarray(qs)
    if np.any(q_array < kin.lower - 1e-7) or np.any(q_array > kin.upper + 1e-7):
        raise RuntimeError("Trajectory exceeds official URDF joint limits")
    speed = np.abs(np.diff(q_array, axis=0)) / cfg["physics_dt"]
    if np.any(speed > cfg["joint_speed"] * 1.01):
        raise RuntimeError("Trajectory exceeds the configured joint speed")
    return Trajectory(np.arange(len(qs)) * cfg["physics_dt"], q_array,
                      np.array(floors), np.array(phases), np.asarray(tips))
