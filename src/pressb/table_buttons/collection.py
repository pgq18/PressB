"""CPU helpers for synchronized scripted tabletop-button demonstrations."""
from __future__ import annotations

from copy import deepcopy
import numpy as np

from .motion import plan_press
from .placement import ReachablePlacement
from .task import COLORS, task_instruction


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def measured_state(joints):
    """Six arm angles plus the full opening, from the two measured fingers."""
    joints = np.asarray(joints, dtype=float)
    if joints.shape != (8,) or not np.isfinite(joints).all():
        raise ValueError("Expected eight finite measured articulation joints")
    return np.r_[joints[:6], joints[6] - joints[7]]


def validate_raw_arrays(frames, physics, *, fps=30, physics_dt=1/120):
    """Check frame/state/action alignment against the actual applied targets."""
    times = np.asarray(frames['sim_time'])
    stride = round(1 / fps / physics_dt)
    if len(times) < 2 or not np.allclose(times, np.arange(len(times))/fps, rtol=0, atol=1e-12):
        raise ValueError("Frame timestamps must start at zero and advance at fps")
    if abs(stride*physics_dt-1/fps) > 1e-12:
        raise ValueError("Frame rate must divide the physical rate")
    if not np.allclose(physics['time_s'], np.arange(len(physics['time_s']))*physics_dt, rtol=0, atol=1e-12):
        raise ValueError("Physical timestamps must include zero at the physical rate")
    if len(physics['time_s']) != (len(times)-1)*stride+1:
        raise ValueError("Physical and camera timelines have different endpoints")
    for key in ('state', 'action'):
        if frames[key].shape != (len(times),7) or not np.isfinite(frames[key]).all():
            raise ValueError(f"Invalid {key} dimensions or values")
    expected = np.array([measured_state(q) for q in physics['q_actual'][::stride]])
    if not np.allclose(frames['state'], expected, rtol=0, atol=1e-12):
        raise ValueError("Measured states do not match camera-boundary physics")
    for index in range(len(times)-1):
        applied = physics['q_target'][index*stride+1:(index+1)*stride+1]
        expected_q = np.r_[frames['action'][index,:6], frames['action'][index,6]/2, -frames['action'][index,6]/2]
        if not np.allclose(applied, expected_q, rtol=0, atol=1e-12):
            raise ValueError("Action is not the target applied over the next control interval")
    if not np.array_equal(frames['action'][-1], frames['action'][-2]):
        raise ValueError("Terminal action must repeat the preceding applied target")


class LayoutPlanner:
    """Cache expensive reachability setup; each seed remains an independent layout."""
    def __init__(self, kin, config):
        self.kin = kin
        self.config = deepcopy(config)
        self.placements = {}

    def placement(self, profile):
        if profile not in ('near', 'medium', 'far'):
            raise ValueError("Unknown spacing profile")
        if profile not in self.placements:
            cfg = deepcopy(self.config)
            cfg['position_sampling']['spacing_profile'] = profile
            self.placements[profile] = ReachablePlacement(self.kin, cfg)
        return self.placements[profile]

    def sample(self, profile, seed):
        return serializable(self.placement(profile).sample(seed))

    def prepare(self, profile, seed, target, layout):
        placement = self.placement(profile)
        # Serialized layouts must retain the configured color route. In
        # particular, an old shared layout cannot silently bypass a requested
        # detour, and a new detour cannot leak into a shared-mode experiment.
        for color in COLORS:
            placement.motion_templates.validate(layout['approaches'][color], color)
        if placement.motion_templates.enabled:
            accepted, diagnostic = placement.validate_layout(layout['positions'], layout['approaches'])
            if not accepted:
                raise ValueError(f'Color-template layout preflight failed: {diagnostic}')
        cfg = deepcopy(self.config)
        cfg['position_sampling']['spacing_profile'] = profile
        cfg['task_color'] = target
        cfg['task_instruction'] = task_instruction(target)
        for color in COLORS:
            cfg['buttons'][color]['center_xy'] = layout['positions'][color]
        cfg['press_approaches'] = layout['approaches']
        cfg['sampling_domain'] = layout['domain']
        geometry = cfg['button_geometry']
        top = geometry['base_height_m'] + geometry['cap_height_m']
        initial = np.r_[cfg['initial_joint_positions_rad'], cfg['initial_gripper_width_m']]
        times, commands, report = plan_press(self.kin, initial,
            np.r_[layout['positions'][target],top], fps=cfg['fps'], approach=layout['approaches'][target])
        dense_times = np.arange(round(times[-1]/cfg['physics_dt'])+1)*cfg['physics_dt']
        dense = np.column_stack([np.interp(dense_times,times,commands[:,j]) for j in range(6)])
        clearance = placement.clearance
        ok, diagnostic = clearance.check_path(dense, initial[6])
        if not ok:
            raise ValueError(f'Dense arm/table preflight failed: {diagnostic}')
        ok, button_diagnostic = clearance.check_button_path(dense,layout['positions'],target,initial[6])
        if not ok:
            raise ValueError(f'Dense button preflight failed: {button_diagnostic}')
        report.update(dense_samples=len(dense_times),dense_physics_dt=cfg['physics_dt'],
                      arm_table_clearance=diagnostic,button_clearance=button_diagnostic,
                      command_policy='30 Hz zero-order hold, four physics steps per interval')
        return cfg,times,commands,serializable(report)
