import numpy as np
import pytest
from pressb.runtime.demonstrations import perturb_plan


@pytest.mark.parametrize('approach',['pregrasp','approach_open'])
def test_smooth_noise_changes_executed_arm_targets_but_preserves_contact_and_gripper(approach):
    phases=[approach]*60+['descend']*30+['close']*30+['transfer']*60+['release']*30+['settled']*30
    actions=np.tile([.1,.2,-.3,.1,.4,.1,.06],(len(phases),1))
    source=dict(actions=actions.tolist(),phases=phases,fps=30)
    result=perturb_plan(source,5,amplitude_deg=.12)
    changed=np.array(result['actions'])-actions
    assert np.max(np.abs(changed[:,:6]))>0
    assert np.max(np.abs(changed[:,:6]))<=np.deg2rad(.12)+1e-12
    np.testing.assert_array_equal(changed[:,6],0.)
    np.testing.assert_array_equal(changed[60:120],0.)
    np.testing.assert_array_equal(changed[180:],0.)
    np.testing.assert_allclose(changed[[0,59,120,179]],0.,atol=1e-15)
    np.testing.assert_array_equal(source['actions'],actions)
    assert result==perturb_plan(source,5,amplitude_deg=.12)
    assert result['actions']!=perturb_plan(source,6,amplitude_deg=.12)['actions']


@pytest.mark.parametrize('amplitude',[float('nan'),-1.])
def test_noise_rejects_invalid_amplitude(amplitude):
    with pytest.raises(ValueError):perturb_plan(dict(actions=[[0]*7]*4,phases=['transfer']*4,fps=30),1,amplitude_deg=amplitude)


def test_noise_scales_smoothly_near_limits_instead_of_rejecting_nominal_trajectory():
    actions=np.zeros((60,7));actions[:,0]=.9999;actions[:,1]=-.9999
    plan=dict(actions=actions.tolist(),phases=['transfer']*60,fps=30)
    result=perturb_plan(plan,7,joint_lower_rad=[-1]*6,joint_upper_rad=[1]*6)
    q=np.asarray(result['actions'])[:,:6]
    assert np.max(q)<=1. and np.min(q)>=-1.
    assert np.max(np.abs(q[:,2:]))>0
    np.testing.assert_allclose(q[[0,-1]],actions[[0,-1],:6],atol=1e-12)
