"""Optional color-specific transit routes; final press geometry stays task-driven."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import numpy as np
from scipy.spatial.transform import Rotation

COLORS = ('red', 'yellow', 'blue')
DEFAULT_WAYPOINTS = {'red': [.26, -.20, .24], 'yellow': [.30, 0., .36], 'blue': [.26, .20, .24]}


def quintic(u):
    u = np.clip(u, 0., 1.)
    return u*u*u*(10+u*(-15+6*u))


class ColorMotionTemplates:
    """Select target-relative routes or reproduce historical base-frame detours.

    All-color validation rejects IK, clearance or trajectory separation failures.
    Defaults are deliberately absent from the legacy shared scene configuration.
    """
    def __init__(self, kin, cfg):
        raw = cfg.get('motion_templates', {})
        if not isinstance(raw, dict):
            raise ValueError('motion_templates must be a mapping')
        self._target_relative = None
        if raw.get('mode') == 'target_relative_v2':
            from .target_relative_motion import TargetRelativeTemplates
            self._target_relative = TargetRelativeTemplates(kin, cfg)
            self.enabled = True
            self.options = self._target_relative.options
            self.signature = self._target_relative.signature
            return
        allowed = {'mode', 'waypoints_base_m', 'pitch_deg', 'leg_duration_s', 'min_transit_separation_m'}
        if set(raw)-allowed:
            raise ValueError(f'Unknown motion_templates options: {sorted(set(raw)-allowed)}')
        mode = raw.get('mode', 'shared')
        if mode not in ('shared', 'color_detours_v1'):
            raise ValueError('motion_templates.mode must be shared, color_detours_v1 or target_relative_v2')
        self.enabled = mode == 'color_detours_v1'
        self.kin = kin
        self._joints = {}
        self.initial = np.asarray(cfg['initial_joint_positions_rad'], dtype=float)
        if not self.enabled:
            if set(raw)-{'mode'}:
                raise ValueError('Detour parameters require color_detours_v1')
            self.options = {'mode': 'shared'}
            return
        waypoints = deepcopy(raw.get('waypoints_base_m', DEFAULT_WAYPOINTS))
        if set(waypoints) != set(COLORS):
            raise ValueError('Specify exactly red/yellow/blue waypoints')
        for color, xyz in waypoints.items():
            xyz = np.asarray(xyz, dtype=float)
            if xyz.shape != (3,) or not np.isfinite(xyz).all() or xyz[2] <= .12:
                raise ValueError(f'Invalid elevated {color} waypoint')
            waypoints[color] = xyz.tolist()
        leg = float(raw.get('leg_duration_s', 2.4))
        pitch = float(raw.get('pitch_deg', 120.))
        separation = float(raw.get('min_transit_separation_m', .10))
        if not np.isfinite([leg,pitch,separation]).all() or leg < 1.8 or not 90 < pitch <= 180 or separation <= 0:
            raise ValueError('Invalid detour timing, orientation or minimum separation')
        if abs(leg*cfg.get('fps',30)-round(leg*cfg.get('fps',30))) > 1e-8:
            raise ValueError('Detour leg duration must lie on the control clock')
        self.options = dict(mode=mode,waypoints_base_m=waypoints,pitch_deg=pitch,
                            leg_duration_s=leg,min_transit_separation_m=separation)
        self.signature = hashlib.sha256(json.dumps(self.options,sort_keys=True).encode()).hexdigest()

    def attach(self, approach, color):
        if self._target_relative is not None:
            return self._target_relative.attach(approach, color)
        if not self.enabled:
            self.validate(approach,color)
            return dict(approach)
        if color not in COLORS:
            raise ValueError('A color is required for color-specific candidate screening')
        if color not in self._joints:
            xyz = np.array(self.options['waypoints_base_m'][color])
            orientation = Rotation.from_euler('z',np.arctan2(xyz[1],xyz[0])).as_matrix() @ Rotation.from_euler('y',self.options['pitch_deg'],degrees=True).as_matrix()
            q = self.kin.solve(xyz,self.initial,orientation)
            self._joints[color] = q.tolist()
        result = dict(approach)
        result['transit_waypoints_rad'] = [self._joints[color].copy()]
        result['motion_template'] = dict(mode='color_detours_v1',color=color,
            waypoint_base_m=self.options['waypoints_base_m'][color],
            leg_duration_s=self.options['leg_duration_s'],config_signature=self.signature)
        return result

    def validate(self, approach, color):
        if self._target_relative is not None:
            return self._target_relative.validate(approach, color)
        metadata = approach.get('motion_template', {})
        if not self.enabled:
            if metadata or approach.get('transit_waypoints_rad') or approach.get('transit_path'):
                raise ValueError('A shared scene cannot reuse a color-detour approach')
            return
        expected = self.attach({k:v for k,v in approach.items() if k not in ('motion_template','transit_waypoints_rad')},color)
        if metadata != expected['motion_template'] or not np.array_equal(
                approach.get('transit_waypoints_rad'),expected['transit_waypoints_rad']):
            raise ValueError('Stale or mismatched color motion template in layout')

    def diversity(self, initial_q, approaches):
        if self._target_relative is not None:
            return self._target_relative.diversity(initial_q, approaches)
        if not self.enabled:
            return dict(accepted=True,mode='shared')
        traces={}
        for color in COLORS:
            self.validate(approaches[color],color)
            knots=np.vstack([initial_q,approaches[color]['transit_waypoints_rad'],approaches[color]['q_hover']])
            samples=[]
            for i in range(len(knots)-1):
                blend=quintic(np.linspace(0.,1.,61))
                q=knots[i]+blend[:,None]*(knots[i+1]-knots[i])
                samples.extend(self.kin.fk(row)[:3,3] for row in q[(0 if i==0 else 1):])
            traces[color]=np.asarray(samples)
        distances={f'{a}_{b}':float(np.linalg.norm(traces[a]-traces[b],axis=1).mean())
                   for i,a in enumerate(COLORS) for b in COLORS[i+1:]}
        threshold=self.options['min_transit_separation_m']
        return dict(accepted=min(distances.values())>=threshold,mode='color_detours_v1',
                    metric='mean synchronous FK tip distance during moving transit,121 points; metres',
                    pairwise_mean_tip_distance_m=distances,min_transit_separation_m=threshold)
