"""Shared scripted press motion for workspace screening and Isaac validation."""
from __future__ import annotations

import numpy as np


def screening_path(initial_q, approach, *, max_joint_step=.035):
    """Sample the same joint transit and vertical approach used by the demo.

    This is a discrete geometric check, not continuous collision detection.
    Include the joint-interpolated retreat used after contact feedback holds
    the descent; the nominal Cartesian return follows the checked approach.
    """
    initial_q = np.asarray(initial_q, dtype=float)[:6]
    knots = np.vstack([initial_q, approach['q_hover'], approach['q_path'], approach['q_hover']])
    samples = [knots[0]]
    for a, b in zip(knots[:-1], knots[1:]):
        count = max(1, int(np.ceil(np.max(np.abs(b-a))/max_joint_step)))
        samples.extend(a+(b-a)*t for t in np.linspace(0., 1., count+1)[1:])
    return np.asarray(samples)


def plan_press(kin, initial, top, *, fps=30, approach=None):
    """Quintic transit and Cartesian press with a selected feasible orientation."""
    from scipy.spatial.transform import Rotation

    initial, top = np.asarray(initial), np.asarray(top)
    if approach is None:
        # Preserve the initial demonstration when called without a workspace.
        orientation = (Rotation.from_euler('z', np.arctan2(top[1], top[0])).as_matrix()
                       @ Rotation.from_euler('y', 160, degrees=True).as_matrix())
        hover, near, press = (np.r_[top[:2], z] for z in (.12, top[2]+.018, top[2]-.006))
        q_hover = kin.solve(hover, initial[:6], orientation)
        pitch = 160.
    else:
        orientation = np.asarray(approach['orientation'])
        hover, near, press = [np.asarray(approach[key]) for key in ('hover', 'near', 'press')]
        q_hover = np.asarray(approach['q_hover'])
        pitch = float(approach['pitch_deg'])
        if not np.allclose(press[:2], top[:2], atol=1e-8):
            raise ValueError('Selected approach belongs to a different button')
    times, commands = [], []
    q_seed = q_hover.copy()
    schedule = [(0., .4, 'joint', initial[:6], initial[:6]),
                (.4, 2.2, 'joint', initial[:6], q_hover),
                (2.2, 3.1, 'cartesian', hover, near),
                (3.1, 4.1, 'cartesian', near, press),
                (4.1, 4.45, 'cartesian', press, press),
                (4.45, 5.55, 'cartesian', press, hover),
                (5.55, 7.35, 'joint', q_hover, initial[:6]),
                (7.35, 7.8, 'joint', initial[:6], initial[:6])]
    for seconds in np.arange(round(schedule[-1][1]*fps)+1)/fps:
        start, end, kind, a, b = next((row for row in schedule if seconds <= row[1]+1e-9), schedule[-1])
        t = np.clip((seconds-start)/(end-start), 0, 1)
        blend = t*t*t*(10+t*(-15+6*t))
        value = np.asarray(a)+(np.asarray(b)-a)*blend
        if kind == 'cartesian':
            q_seed = kin.solve(value, q_seed, orientation)
            q = q_seed.copy()
        else:
            q = value
        times.append(seconds)
        commands.append(np.r_[q, initial[6]])
    commands = np.asarray(commands)
    max_velocity = np.max(np.abs(np.diff(commands[:, :6], axis=0)), axis=0)*fps
    if np.any(max_velocity > kin.velocity_limits):
        raise ValueError(f'Demo exceeds URDF velocity limits: {max_velocity}')
    return np.asarray(times), commands, dict(
        hover_tcp_base_m=hover.tolist(), press_tcp_base_m=press.tolist(),
        approach_pitch_deg=pitch, max_joint_velocity_rad_s=max_velocity.tolist(),
        method='workspace-selected bounded IK and quintic interpolation')
