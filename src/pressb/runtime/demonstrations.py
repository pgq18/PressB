"""Shared, time-aligned raw demonstrations and bounded command perturbations."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import numpy as np

from .recording import Video, write


def perturb_plan(plan, seed, *, amplitude_deg=.12, joint_lower_rad=None, joint_upper_rad=None):
    """Smooth arm-only noise during free-space approach/transfer, zero at ends.

    Grasp, descent, placement, release and gripper commands stay exact. Noise
    changes the actually dispatched commands; it is never added only to labels.
    """
    if not np.isfinite(amplitude_deg) or amplitude_deg < 0:
        raise ValueError("Noise amplitude must be finite and nonnegative")
    result=deepcopy(plan)
    actions=np.asarray(plan['actions'],float)
    noise=np.zeros_like(actions)
    bounds=None
    if joint_lower_rad is not None or joint_upper_rad is not None:
        lower,upper=np.asarray(joint_lower_rad,float),np.asarray(joint_upper_rad,float)
        if lower.shape!=(6,) or upper.shape!=(6,) or not np.isfinite(np.r_[lower,upper]).all() or np.any(lower>=upper):
            raise ValueError('Noise bounds require six ordered finite joint limits')
        bounds=(lower,upper)
    rng=np.random.default_rng(seed)
    phases=plan['phases']
    first=0
    while first<len(actions):
        last=first+1
        while last<len(actions) and phases[last]==phases[first]:last+=1
        if phases[first].split(':')[-1] in ('pregrasp','approach_open','transfer') and last-first>=4:
            u=np.linspace(0.,1.,last-first)[:,None]
            smooth=np.sin(np.pi*u)**2*np.sin(2*np.pi*(u*rng.uniform(.5,1.5,(1,6))+rng.uniform(0,1,(1,6))))
            smooth/=np.maximum(1.,np.max(np.abs(smooth),axis=0,keepdims=True))
            noise[first:last,:6]=np.deg2rad(amplitude_deg)*smooth
            if bounds is not None:
                # One scale per phase and joint preserves smoothness, even for
                # trajectories near a limit. Do not clip individual samples.
                segment=noise[first:last,:6]
                room=np.where(segment>0,bounds[1]-actions[first:last,:6],actions[first:last,:6]-bounds[0])
                ratios=np.divide(np.maximum(0.,room),np.abs(segment),out=np.full_like(segment,np.inf),where=np.abs(segment)>1e-12)
                segment*=np.minimum(1.,np.min(ratios,axis=0))[None,:]
        first=last
    result['actions']=(actions+noise).tolist()
    result['noise']=dict(kind='smooth_bounded_joint_command',seed=int(seed),amplitude_deg=amplitude_deg,
        phases=['pregrasp','approach_open','transfer'],gripper_noise=False,
        bounds_policy='Smooth phase-wise joint scaling to existing joint limits' if bounds is not None else 'Unscaled',
        max_actual_degrees=float(np.rad2deg(np.max(np.abs(noise[:,:6])))),
        max_rate_deg_s=float(np.rad2deg(np.max(np.abs(np.diff(noise[:,:6],axis=0))))*plan['fps']),
        waypoint_semantics='Waypoints describe nominal path; actions contain actual perturbed targets')
    return result


class RawEpisode:
    """Record obs_t with the next actually applied action_t, then a terminal row."""
    def __init__(self, directory, *, fps=30.):
        self.directory=Path(directory); self.directory.mkdir(parents=True,exist_ok=False)
        self.fps=float(fps)
        self.videos={key:Video(self.directory/f'{key}.mp4',(640,480),fps) for key in ('up','wrist')}
        self.states=[]; self.actions=[]
        self.trace=(self.directory/'physics.jsonl').open('w')
        self.closed=False

    def frame(self, state, action, images):
        state,action=np.asarray(state,float),np.asarray(action,float)
        if state.shape!=(7,) or action.shape!=(7,) or not np.isfinite(np.r_[state,action]).all():
            raise ValueError('Expected finite seven-coordinate state/action')
        for name,pipe in self.videos.items():
            rgb=np.asarray(images[name])
            if rgb.dtype!=np.uint8 or rgb.shape!=(480,640,3):raise ValueError('Expected 640x480 RGB')
            pipe.add(np.ascontiguousarray(rgb))
        self.states.append(state.copy());self.actions.append(action.copy())

    def physical(self, row):
        self.trace.write(json.dumps(row,allow_nan=False)+'\n')

    def close(self):
        if self.closed:return
        self.closed=True
        self.trace.close()
        errors=[]
        for pipe in self.videos.values():
            try:pipe.close()
            except Exception as exc:errors.append(exc)
        np.savez_compressed(self.directory/'data.npz',sim_time=np.arange(len(self.states))/self.fps,
                            state=np.asarray(self.states),action=np.asarray(self.actions))
        if errors:raise RuntimeError('One or more episode video encoders failed') from errors[0]

    def commit(self, destination, metadata):
        self.close()
        if len(self.actions)<2 or not np.array_equal(self.actions[-1],self.actions[-2]):
            raise ValueError('Terminal frame must repeat previous executed action')
        metadata={**metadata, 'frame_count':len(self.states),'fps':self.fps,
            'state_units':['rad']*6+['m'],'action_units':['rad']*6+['m'],
            'gripper_representation':'full_width','terminal_action':'repeat_previous_interval_target',
            'camera_resolution':[640,480], 'initial_state_measured_rad_m':self.states[0].tolist(),
            'alignment':'observation_t and upcoming executed action_t; last row holds previous action'}
        write(self.directory/'metadata.json',metadata)
        self.directory.rename(destination)
