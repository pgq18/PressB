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
    waypoints = approach.get('transit_waypoints_rad', [])
    if 'transit_path' in approach:
        if approach.get('motion_template', {}).get('mode') != 'target_relative_v2':
            raise ValueError('A sampled transit requires its target-relative template')
        knots = np.vstack([approach['transit_path']['joints_rad'], approach['q_path'][-1],
                           approach['q_hover'], initial_q])
    else:
        knots = np.vstack([initial_q, *waypoints, approach['q_hover'], approach['q_path'], approach['q_hover'], *([initial_q] if waypoints else [])])
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
    via = [] if approach is None else approach.get('transit_waypoints_rad', [])
    template = {} if approach is None else approach.get('motion_template', {})
    sampled = None if approach is None else approach.get('transit_path')
    if sampled is not None:
        if template.get('mode') != 'target_relative_v2' or via:
            raise ValueError('Expected one declared target-relative sampled path')
        sample_times = np.asarray(sampled['time_s'], dtype=float)
        sample_joints = np.asarray(sampled['joints_rad'], dtype=float)
        duration = float(template['transit_duration_s'])
        if (sample_times.ndim != 1 or len(sample_times) < 2
                or sample_joints.shape != (len(sample_times), 6)
                or not np.isfinite(sample_times).all() or not np.isfinite(sample_joints).all()
                or np.any(np.diff(sample_times) <= 0) or sample_times[0] != 0
                or not np.isfinite(duration) or duration < 2.7 or abs(sample_times[-1]-duration) > 1e-9
                or not np.allclose(sample_joints[0], initial[:6], rtol=0., atol=1e-10)
                or np.any(sample_joints < kin.lower) or np.any(sample_joints > kin.upper)):
            raise ValueError('Invalid target-relative sampled path or initial pose')
        if not np.allclose(kin.fk(sample_joints[-1])[:3, 3], near, rtol=0., atol=5e-4):
            raise ValueError('Target-relative transit must end at the selected near pose')
        shift = duration-2.7
    elif via:
        if template.get('mode') != 'color_detours_v1' or np.asarray(via).shape != (1,6):
            raise ValueError('Expected one declared color-detour waypoint')
        if not np.isfinite(via).all() or np.any(np.asarray(via)<kin.lower) or np.any(np.asarray(via)>kin.upper):
            raise ValueError('Detour joint waypoint exceeds finite joint limits')
        leg = float(template['leg_duration_s'])
        if not np.isfinite(leg) or leg < 1.8:
            raise ValueError('Invalid color-detour leg duration')
        shift = 2*leg - 1.8
    else:
        if template:
            raise ValueError('Declared template lacks its detour waypoint')
        shift = 0.
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
    if sampled is not None:
        tail = [(a+shift,b+shift,kind,x,y) for a,b,kind,x,y in schedule[3:]]
        schedule = [schedule[0],(.4,.4+duration,'sampled',None,None),*tail]
    elif via:
        tail = [(a+shift,b+shift,kind,x,y) for a,b,kind,x,y in schedule[2:]]
        schedule = [schedule[0],(.4,.4+leg,'joint',initial[:6],np.asarray(via[0])),
                    (.4+leg,.4+2*leg,'joint',np.asarray(via[0]),q_hover),*tail]
    for seconds in np.arange(round(schedule[-1][1]*fps)+1)/fps:
        start, end, kind, a, b = next((row for row in schedule if seconds <= row[1]+1e-9), schedule[-1])
        t = np.clip((seconds-start)/(end-start), 0, 1)
        blend = t*t*t*(10+t*(-15+6*t))
        if kind == 'sampled':
            q = np.array([np.interp(seconds-start, sample_times, sample_joints[:,j]) for j in range(6)])
            q_seed = q.copy()
        else:
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
        method='workspace-selected bounded IK and quintic interpolation',
        motion_template=template or {'mode':'shared'},
        phase_times=dict(transit_start=.4,
                         transit_end=(.4+duration if sampled is not None else 2.2+shift),
                         approach_start=(.4+duration*template['time_fractions'][-2] if sampled is not None else 2.2+shift),
                         press_start=3.1+shift,release_start=4.45+shift,return_hover_end=5.55+shift))
