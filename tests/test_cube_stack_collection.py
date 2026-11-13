"""Settled-pose planning and acceptance for successful stack datasets."""
from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pressb.tasks.cube_stack.collection import collection_config,plan_episode,assess_episode
from pressb.tasks.cube_stack.configuration import load_config
from pressb.tasks.cube_stack.task import footprint


@pytest.fixture(scope='module')
def inputs():
    cfg=load_config(collection_config())
    mount=cfg['scene']['robot_base_height_above_table_m']
    base=np.array([0.,0.,cfg['scene']['table_height']+mount])
    snapshot={'objects':{}}
    reset={'objects':{}}
    for name,obj in cfg['cubes'].items():
        p=np.r_[np.array(obj['center_xy'])+[.001,0.],obj['edge_m']/2-mount]+base
        q=Rotation.from_euler('z',obj['yaw_deg'],degrees=True).as_quat()[[3,0,1,2]]
        snapshot['objects'][name]=dict(position_world=p.tolist(),orientation_wxyz=q.tolist())
        reset['objects'][name]=dict(center_xy=obj['center_xy'],yaw_deg=obj['yaw_deg'])
    return cfg,reset,snapshot,base


@pytest.fixture(scope='module')
def plan(inputs):
    return plan_episode(*inputs)


def test_configuration_samples_position_and_yaw_and_preserves_outcome_rules():
    raw=collection_config(); cfg=load_config(raw)
    assert cfg['randomization']['enabled']
    assert cfg['randomization']['yaw_range_deg']==[-45.,45.]
    assert cfg['randomization']['x_range_m'][1]-cfg['randomization']['x_range_m'][0]>=.1
    assert cfg['randomization']['y_range_m'][1]-cfg['randomization']['y_range_m'][0]>=.25
    assert cfg['terminate_on_success'] is False
    assert 'min_xy_overlap_fraction' not in cfg['outcome']
    assert cfg['outcome']['min_grasp_force_n']==.02


def test_planning_uses_observed_pose_and_equivalent_grasp_without_mutating_inputs(inputs,plan):
    cfg,reset,snapshot,base=inputs
    assert plan['planning_pose_source']=='measured_settled_xy_yaw'
    observed=np.array(snapshot['objects']['red']['position_world'])-base
    np.testing.assert_allclose(plan['source']['center_xy'],observed[:2])
    assert plan['source']['center_xy']!=reset['objects']['red']['center_xy']
    turn=plan['grasp_symmetry_quarter_turns']
    assert plan['grasp_equivalent_cube_yaw_deg']==pytest.approx(plan['source_observed_yaw_deg']+turn*90)
    before=footprint([0.,0.],.029,plan['source_observed_yaw_deg'])
    after=footprint([0.,0.],.029,plan['grasp_equivalent_cube_yaw_deg'])
    assert all(np.min(np.linalg.norm(after-p,axis=1))<1e-12 for p in before)
    assert cfg['cubes']['red']['center_xy']==reset['objects']['red']['center_xy']
    assert plan['trajectory_profile']=='compact'
    assert max(row[6] for row in plan['actions'])==pytest.approx(.100)
    assert plan['phases'][-1]=='retreat'


def test_environment_translation_preserves_identical_robot_plan(inputs,plan):
    cfg,reset,snapshot,base=inputs
    moved=deepcopy(snapshot);shift=np.array([6.,-6.,0.])
    for state in moved['objects'].values():state['position_world']=(np.array(state['position_world'])+shift).tolist()
    peer=plan_episode(cfg,reset,moved,base+shift)
    np.testing.assert_allclose(peer['actions'],plan['actions'],atol=1e-9)


@pytest.mark.parametrize('fault',['tilt','airborne','nonfinite','bad_quaternion'])
def test_unsettled_or_malformed_observation_is_rejected(inputs,fault):
    cfg,reset,snapshot,base=inputs
    bad=deepcopy(snapshot);red=bad['objects']['red']
    if fault=='tilt':red['orientation_wxyz']=Rotation.from_euler('x',12,degrees=True).as_quat()[[3,0,1,2]].tolist()
    elif fault=='airborne':red['position_world'][2]+=.02
    elif fault=='nonfinite':red['position_world'][0]=float('nan')
    else:red['orientation_wxyz']=[0,0,0,0]
    with pytest.raises(ValueError):plan_episode(cfg,reset,bad,base)


def test_legacy_acceptance_requires_final_physical_success_and_complete_recording(inputs):
    plan=plan_episode(*inputs,trajectory_profile='legacy')
    rows=[]
    for z in [.7745,.86]+[.8045]*20:
        rows.append(dict(objects={'red':dict(position_world=[.3,.1,z],orientation_wxyz=[1,0,0,0]),
                                  'blue':dict(position_world=[.3,.1,.775],orientation_wxyz=[1,0,0,0])},
            state=[*plan['expected_initial_state'][:6],.06],
            info=dict(success=True,physical_outcome=dict(success=True,grasped_and_lifted=True,dropped=False)),
            phase='settled',action_index=len(plan['actions'])-1))
    assert assess_episode(rows,plan)['passed']
    for key in ['success','grasped_and_lifted']:
        changed=deepcopy(rows);changed[-1]['info']['physical_outcome'][key]=False
        assert not assess_episode(changed,plan)['passed']
    changed=deepcopy(rows);changed[-1]['phase']='release'
    assert not assess_episode(changed,plan)['passed']


def _compact_success_history(plan):
    rows=[]
    for index,z in enumerate([.7745,.86,.8045]):
        rows.append(dict(objects={'red':dict(position_world=[.3,.1,z],orientation_wxyz=[1,0,0,0]),
                                  'blue':dict(position_world=[.3,.1,.775],orientation_wxyz=[1,0,0,0])},
            state=list(plan['actions'][-1]),
            info=dict(success=True,physical_outcome=dict(success=True,grasped_and_lifted=True,dropped=False)),
            phase=plan['phases'][-1],action_index=len(plan['actions'])-3+index))
    for index in range(18):
        row=deepcopy(rows[-1]);row.update(phase='validation',validation_index=index,recorded=False)
        rows.append(row)
    return rows


def test_compact_acceptance_uses_unrecorded_validation_and_final_retreat_target(plan):
    rows=_compact_success_history(plan)
    audit=assess_episode(rows,plan)
    assert audit['passed']
    assert audit['full_plan_executed'] and audit['terminal_validation_complete']
    assert audit['terminal_validation_frames']==audit['required_validation_frames']==18
    assert audit['max_final_target_joint_error_rad']==0.
    assert audit['max_home_joint_error_rad']>.03
    assert not assess_episode(rows[:-1],plan)['passed']
    assert not assess_episode(rows[:3],plan)['passed']


@pytest.mark.parametrize('fault',['unfinished_motion','wrong_validation_target','missing_validation_step',
    'early_validation','not_at_retreat','dropped','not_grasped','unstable','truncated'])
def test_compact_acceptance_rejects_incomplete_or_physically_invalid_episode(plan,fault):
    rows=_compact_success_history(plan)
    if fault=='unfinished_motion':rows[2]['action_index']-=1
    elif fault=='wrong_validation_target':rows[-1]['action_index']-=1
    elif fault=='missing_validation_step':rows[-1]['validation_index']+=1
    elif fault=='early_validation':rows[2]['phase']='release'
    elif fault=='not_at_retreat':rows[-1]['state'][0]+=.031
    elif fault=='dropped':rows[-1]['info']['physical_outcome']['dropped']=True
    elif fault=='not_grasped':rows[-1]['info']['physical_outcome']['grasped_and_lifted']=False
    elif fault=='unstable':rows[-1]['objects']['red']['position_world'][0]+=.002
    else:rows[-1]['truncated']=True
    assert not assess_episode(rows,plan)['passed']
