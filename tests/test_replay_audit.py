"""Negative replay evidence must fail even when runner metadata says success."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np
import pytest

av = pytest.importorskip('av')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_replay import (FULL_CYCLE_END, PRESS_ONLY_END, audit_physics, audit_video,
                          normalized_rotations, overview_phases, recompute_lights, rotation_errors)
from prepare_replay import episode_end_policy
from replay_dataset import terminal_condition_met


class FixedTestFK:
    lower = np.full(6, -1.)
    upper = np.full(6, 1.)

    def batch(self, q):
        return np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()


def evidence():
    cfg = dict(home_q=[0.]*6, robot_urdf='test.urdf', robot_base_x=0., robot_base_y=0.,
               table_height=0., physics_dt=1/120, press_threshold=.0015,
               release_threshold=.0008, tip_offset=.24, gripper_joint_positions_m=[.004,-.004],
               home_tolerance_rad=.025)
    count = 13
    fingers = np.broadcast_to([.004,-.004], (count,2)).copy()
    state = np.broadcast_to([0.,0.,0.,1.,0.,0.,0.,.008], (count,8)).copy()
    physics = dict(q_actual=np.zeros((count,6)), q_command=np.zeros((count,6)),
                   gripper_actual=fingers.copy(), gripper_command=fingers.copy(),
                   button_travel=np.zeros((count,12)), contact_force=np.zeros((count,12)),
                   lights=np.zeros((count,12), dtype=np.uint8), tcp_state=state,
                   tip_position_world=np.broadcast_to([0.,0.,.24-.1358],(count,3)).copy(),
                   time=np.arange(count)/120)
    physics['button_travel'][4:10,0] = .002
    physics['contact_force'][4:10,0] = .1
    physics['lights'][4:10,0] = 1
    source = dict(action=state[::4].copy(), state=state[::4].copy(), sim_time=np.arange(4)/30)
    source_metadata = dict(source_episode_id=0, floor=24, task='Press 24 floor.')
    metadata = dict(**source_metadata, config=deepcopy(cfg), fps=30, physics_hz=120,
                    num_frames=4, capture_physics_indices=[0,4,8,12], success=True,
                    robot_base_world_m=[0.,0.,0.], env_offset_m=[0.,0.,0.], unexpected_collisions=[],
                    initial_joint_velocity_rad_s=[0.]*6,
                    events=[dict(type='pressed',floor=24,physics_index=4),
                            dict(type='released',floor=24,physics_index=10)])
    return physics,metadata,source,source_metadata,cfg,FixedTestFK()


def test_consistent_evidence_passes_without_trusting_runner_success():
    args = evidence()
    args[1]['success'] = False
    report = audit_physics(*args)
    assert report['success'] and report['errors'] == []
    assert report['final_home_error_rad'] == 0


def press_only_evidence():
    args = evidence()
    p, m, _, source_metadata, _, _ = args
    for target in (m, source_metadata):
        target['episode_end'] = PRESS_ONLY_END
        target['episode_end_source_export_manifest_sha256'] = 'a' * 64
    p['button_travel'][:] = 0.
    p['contact_force'][:] = 0.
    p['lights'][:] = 0
    p['button_travel'][10:, 0] = .002
    p['contact_force'][10:, 0] = .1
    p['lights'][10:, 0] = 1
    # The end posture deliberately differs from home; a successful press-only
    # episode must not acquire a new requirement to retract or release.
    p['q_actual'][:, 0] = np.linspace(0., .2, len(p['q_actual']))
    p['q_command'][:] = p['q_actual']
    m['events'] = [dict(type='pressed', floor=24, physics_index=10)]
    return args


def test_press_only_success_ends_lit_away_from_home_without_release():
    args = press_only_evidence()
    report = audit_physics(*args)
    assert report['success'] and report['errors'] == []
    assert report['final_home_error_rad'] == .2
    assert report['sampled_lit_frame_indices'] == [3]
    assert overview_phases(report) == dict(initial=0, approach=2, pressed=3, terminal=3)
    p, m, _, _, cfg, _ = args
    assert terminal_condition_met(PRESS_ONLY_END, 24, [24], [], {24}, .2, p['button_travel'][-1], cfg)
    assert not terminal_condition_met(FULL_CYCLE_END, 24, [24], [], {24}, .2, p['button_travel'][-1], cfg)


@pytest.mark.parametrize('corruption,expected', [
    ('missing_press', 'terminal observation'), ('wrong_floor', 'no other floor'),
    ('false_light_label', 'contact/travel hysteresis'), ('early_release', 'remain pressed'),
    ('changed_policy', 'termination differs'), ('changed_export_hash', 'export provenance'),
])
def test_press_only_corrupt_or_wrong_terminal_evidence_is_rejected(corruption, expected):
    args = press_only_evidence()
    p, m, _, _, cfg, _ = args
    if corruption == 'missing_press':
        p['button_travel'][:] = 0.; p['contact_force'][:] = 0.; p['lights'][:] = 0
        m['events'] = []
        assert not terminal_condition_met(PRESS_ONLY_END, 24, [], [], set(), .2, p['button_travel'][-1], cfg)
    elif corruption == 'wrong_floor':
        for key in ('button_travel', 'contact_force', 'lights'):
            p[key][:, 1] = p[key][:, 0]; p[key][:, 0] = 0
        m['events'][0]['floor'] = 25
        assert not terminal_condition_met(PRESS_ONLY_END, 24, [25], [], {25}, .2, p['button_travel'][-1], cfg)
    elif corruption == 'false_light_label':
        p['lights'][-1, 0] = 0
    elif corruption == 'early_release':
        p['button_travel'][-1, 0] = 0.; p['contact_force'][-1, 0] = 0.; p['lights'][-1, 0] = 0
        m['events'].append(dict(type='released', floor=24, physics_index=12))
    elif corruption == 'changed_policy':
        m['episode_end'] = FULL_CYCLE_END
    elif corruption == 'changed_export_hash':
        m['episode_end_source_export_manifest_sha256'] = 'b' * 64
    report = audit_physics(*args)
    assert not report['success']
    assert any(expected in error for error in report['errors'])


def test_source_episode_end_is_explicit_or_legacy_and_unknown_values_fail():
    assert episode_end_policy({}) == FULL_CYCLE_END
    assert episode_end_policy({'episode_end': PRESS_ONLY_END}) == PRESS_ONLY_END
    with pytest.raises(ValueError, match='Unsupported recorded episode_end'):
        episode_end_policy({'episode_end': 'anything_counts_as_success'})
    args = press_only_evidence()
    del args[3]['episode_end_source_export_manifest_sha256']
    with pytest.raises(ValueError, match='export manifest SHA256'):
        audit_physics(*args)


def test_force_and_travel_are_both_required_and_release_has_hysteresis():
    travel = np.zeros((5,12)); force = np.zeros((5,12))
    travel[:,0] = [.0015,.0015,.001,.0009,.0008]
    force[:,0] = [.02,.021,0.,0.,0.]
    lights,events = recompute_lights(travel,force,.0015,.0008)
    assert lights[:,0].tolist() == [0,1,1,1,0]
    assert events == [('pressed',24,1),('released',24,4)]


def test_float32_quaternion_rounding_does_not_create_orientation_error():
    q = np.asarray([[.7071067811865476,0.,0.,.7071067811865476]], dtype=np.float32)
    exact = np.array([[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]])
    assert rotation_errors(exact, normalized_rotations(q)).max() < 1e-7


@pytest.mark.parametrize('corruption,expected', [
    ('force','contact/travel hysteresis'), ('extra_floor','no other floor'),
    ('collision','Unexpected contact'), ('tip','independent URDF FK'),
    ('orientation','full action orientations'), ('tracking','joint tracking'),
    ('initial_velocity','physically settled'),
])
def test_bad_evidence_is_rejected_despite_success_true(corruption,expected):
    args = evidence(); p,m,s,_,_,_ = args
    if corruption == 'force': p['contact_force'][:] = 0.
    elif corruption == 'extra_floor':
        p['button_travel'][5:8,1] = .002; p['contact_force'][5:8,1] = .1; p['lights'][5:8,1] = 1
    elif corruption == 'collision': m['unexpected_collisions'] = [{'body':'table','force_n':1.}]
    elif corruption == 'tip': p['tip_position_world'][6,0] += .006
    elif corruption == 'orientation': s['action'][:,3:7] = [np.cos(.2),0.,0.,np.sin(.2)]
    elif corruption == 'tracking': p['q_actual'][6,0] = .16
    elif corruption == 'initial_velocity': m['initial_joint_velocity_rad_s'][0] = .021
    report = audit_physics(*args)
    assert not report['success']
    assert any(expected in error for error in report['errors'])


def write_video(path, fps=30, count=4):
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=fps)
        stream.width,stream.height,stream.pix_fmt = 640,480,'yuv420p'
        for i in range(count):
            frame = av.VideoFrame.from_ndarray(np.full((480,640,3),30+i*30,dtype=np.uint8),format='rgb24')
            frame.pts,frame.time_base = i,Fraction(1,fps)
            for packet in stream.encode(frame): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)


def test_actual_video_rate_pts_and_full_frame_count(tmp_path):
    path = tmp_path/'camera.mp4'; write_video(path)
    report,images = audit_video(path,4,np.arange(4)/30,{0,3})
    assert report['decoded_frames']==4 and report['pts_match_physics']
    assert set(images)=={0,3}
    with pytest.raises(ValueError,match='expected 5'):
        audit_video(path,5,np.arange(5)/30,{0})
    slow=tmp_path/'slow.mp4';write_video(slow,fps=10)
    with pytest.raises(ValueError,match='Wrong video fps'):
        audit_video(slow,4,np.arange(4)/30,{0})
