"""Dataset task hooks retain randomization and physical success semantics."""
from copy import deepcopy
from collections import Counter

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pressb.tasks.tabletop_cleanup.collection import (
    PAIRS, assess_episode, collection_config, measured_poses, plan_episode, sample_episode,
)
from pressb.tasks.tabletop_cleanup.configuration import DEFAULT_BIN, load_config, validate_layout
from pressb.tasks.tabletop_cleanup.geometry import asset_spec


@pytest.fixture(scope="module")
def config():
    return load_config(collection_config())


def test_success_slots_balance_pairs_and_retry_preserves_pair(config):
    counts=Counter(PAIRS[i % len(PAIRS)] for i in range(500))
    assert sorted(counts.values())==[166,167,167]
    for index in range(6):
        first=sample_episode(config,20261110+index,index)
        retry=sample_episode(config,30261110+index,index)
        assert tuple(first["active_names"])==tuple(retry["active_names"])==PAIRS[index%3]
        assert len(set(first["active_names"]))==2
        assert first["instruction"]==retry["instruction"]
        assert first["objects"]!=retry["objects"]
        validate_layout(config,first["objects"])
    assert config["bin"]==DEFAULT_BIN
    assert config["randomization"]["fixed_subset"] is None


def test_random_sampling_varies_yaw_and_position_without_changing_sizes(config):
    points=[];yaws=[]
    for i in range(24):
        layout=sample_episode(config,7000+i,i)
        for kind,pose in layout["objects"].items():
            points.append(pose["center_xy"]);yaws.append(pose["yaw_deg"])
            assert config["objects"][kind]["scale"]==1.
        validate_layout(config,layout["objects"])
    assert np.ptp(points,axis=0).min()>.05
    assert np.ptp(yaws)>40.


def test_observation_conversion_handles_cloned_environment_offsets():
    q=Rotation.from_euler('xyz',[-9.6,0,14],degrees=True).as_quat()[[3,0,1,2]]
    snapshot={"objects":{"gamepad":dict(active=True,position_world=[6.36,-5.96,.79],orientation_wxyz=q.tolist())}}
    pose=measured_poses(snapshot,[6.,-6.,.766],["gamepad"])["gamepad"]
    np.testing.assert_allclose(pose["position_base_m"],[.36,.04,.024],atol=1e-12)
    np.testing.assert_allclose(pose["orientation_wxyz"],q)
    snapshot['objects']['gamepad']['active']=False
    with pytest.raises(ValueError):measured_poses(snapshot,[6.,-6.,.766],["gamepad"])


@pytest.mark.parametrize('profile',['legacy','compact'])
def test_plan_uses_settled_pose_and_current_centering_without_scene_mutation(config,profile):
    reset=dict(active_names=['gamepad','speaker'],objects={
        'gamepad':dict(center_xy=[.36584942,.03786663],yaw_deg=5.66315),
        'speaker':dict(center_xy=[.371758,-.10155661],yaw_deg=1.3502)})
    poses={'gamepad':dict(active=True,position_world=[.36536324,.04280226,.7890746],
                       orientation_wxyz=Rotation.from_euler('xyz',[-9.593,0,5.6757],degrees=True).as_quat()[[3,0,1,2]].tolist()),
           'speaker':dict(active=True,position_world=[.371758,-.10155661,.77166003],
                       orientation_wxyz=Rotation.from_euler('z',1.3502,degrees=True).as_quat()[[3,0,1,2]].tolist())}
    original=deepcopy(config)
    options={} if profile=='compact' else {'trajectory_profile':profile}
    p=plan_episode(config,reset,dict(objects=poses),[0,0,.766],**options)
    assert config==original
    assert config["scene"]["gripper_max_width_m"] == .100
    assert p['grasps']['gamepad']['pose_source']=='measured_settled'
    assert p['trajectory_profile']==profile
    assert p["grasps"]["gamepad"]["open_width_m"] == (.100 if profile=='compact' else .090)
    clearance = p["grasps"]["gamepad"]["finger_sweep_clearance"]["after_clearance_m"]
    assert min(clearance.values()) > .015
    assert p['grasps']['speaker']['open_width_m']==(.100 if profile=='compact' else .070)
    assert [w['phase'] for w in p['waypoints'] if w['phase'].endswith('return_home')]==(
        [] if profile=='compact' else ['speaker:return_home'])


def _success_history():
    plan=dict(kinds=['speaker','lidar'],fps=30.,expected_initial_state=[0,0,0,0,0,0,.07],actions=[[0]*7]*36)
    history=[]
    flags=dict.fromkeys(['active','grasped_and_lifted','contained','supported','released','low_velocity','no_table_support'],True)
    for i,z in enumerate([.77,.86]+[.774]*34):
        history.append(dict(state=[0,0,0,0,0,0,.07],phase='settled',action_index=i,
            objects={k:dict(position_world=[.35,.2,z],orientation_wxyz=[1,0,0,0]) for k in plan['kinds']},
            info=dict(success=True,physical_outcome=dict(success=True,objects={k:dict(flags) for k in plan['kinds']}))))
    return plan,history


def test_acceptance_requires_both_grasps_complete_release_and_stability():
    plan,history=_success_history()
    assert assess_episode(history,plan)['passed']
    for key in ['grasped_and_lifted','supported','contained','released','no_table_support']:
        bad=deepcopy(history);bad[-1]['info']['physical_outcome']['objects']['lidar'][key]=False
        assert not assess_episode(bad,plan)['passed']
    bad=deepcopy(history);bad[-1]['truncated']=True
    assert not assess_episode(bad,plan)['passed']
    bad=deepcopy(history);bad[-1]['objects']['speaker']['position_world'][0]+=.004
    assert not assess_episode(bad,plan)['passed']
    assert not assess_episode(history[-5:],plan)['passed']


def _compact_success_history():
    plan,history=_success_history()
    target=[.2,.3,-.4,.1,.8,-.2,.100]
    plan.update(trajectory_profile='compact',validation_hold_seconds=.6,
                phases=['lidar:retreat']*len(plan['actions']))
    plan['actions'][-1]=target
    for row in history:
        row['phase']='lidar:retreat';row['state']=list(target)
    for index in range(18):
        row=deepcopy(history[-1]);row.update(phase='validation',validation_index=index,recorded=False)
        history.append(row)
    return plan,history


def test_compact_acceptance_requires_terminal_stability_without_return_home():
    plan,history=_compact_success_history()
    result=assess_episode(history,plan)
    assert result['passed'] and result['terminal_validation_complete']
    assert result['terminal_validation_frames']==result['required_validation_frames']==18
    assert result['max_final_target_joint_error_rad']==0.
    assert result['max_home_joint_error_rad']>.03
    assert not assess_episode(history[:-1],plan)['passed']
    assert not assess_episode(history[:-18],plan)['passed']


@pytest.mark.parametrize('fault',['unfinished_motion','wrong_validation_target','missing_validation_step',
    'early_validation','not_at_retreat','released','contained','supported','no_table_support','grasped_and_lifted','unstable'])
def test_compact_acceptance_retains_physical_and_completion_requirements(fault):
    plan,history=_compact_success_history()
    if fault=='unfinished_motion':history[-19]['action_index']-=1
    elif fault=='wrong_validation_target':history[-1]['action_index']-=1
    elif fault=='missing_validation_step':history[-1]['validation_index']+=1
    elif fault=='early_validation':history[-19]['phase']='lidar:release'
    elif fault=='not_at_retreat':history[-1]['state'][0]+=.031
    elif fault=='unstable':history[-1]['objects']['speaker']['position_world'][0]+=.004
    else:history[-1]['info']['physical_outcome']['objects']['lidar'][fault]=False
    assert not assess_episode(history,plan)['passed']
