"""Meaningful offline recovery and degeneracy checks; no hardware or rendering."""
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from calibrate_playground_multiview import fit_model,unpack,project
from pressb.robots.piper.kinematics import PiperKinematics
import json


def synthetic_problem():
    cfg=json.loads((ROOT/'configs/piper_playground.json').read_text())
    kin=PiperKinematics(ROOT/cfg['robot_urdf'],tip_offset=0)
    prior=np.array(cfg['wrist_camera']['link6_from_camera_usd'])@np.diag([1,-1,-1,1])
    K=np.array([[605.694458,0,320.630768],[0,605.638489,250.773041],[0,0,1.]])
    truth=np.r_[np.deg2rad([.5,-.3,.4]),[.002,-.003,.004],.356,-.104,-.04,.03]
    E,points=unpack(truth,prior)
    qs=np.array([
        [-.185,.473,-.459,-.114,.748,.082],[-.4,.7,-.7,.1,.8,-.1],
        [.05,.85,-.8,-.2,.6,.3],[-.1,1.1,-1.,.2,.8,-.3],
        [-.3,1.4,-1.15,-.1,.7,.2],[.1,1.65,-1.3,.15,.85,.05],
        [-.285,1.9,-1.45,0,1.2,-.245],[-.05,1.25,-1.0,-.3,1.,.25]])
    observations=[]
    for i,q in enumerate(qs):
        B=kin.fk(q);uv=project(K,B,E,points)
        observations.append({'name':str(i),'base_from_link6':B,'corners_uv':uv})
    T=np.array(observations[0]['base_from_link6'])@E
    plane={'base_from_link6':observations[0]['base_from_link6'],
        'normal_cv':(-T[:3,:3].T@np.array([0,0,1])).tolist(),
        'distance_m':float(T[2,3]),'cube_center_cv_m':(T[:3,:3].T@(points.mean(0)-T[:3,3])).tolist(),
        'cube_edge_m':.03}
    initial=np.r_[np.zeros(6),.35,-.10,0,.03]
    return K,prior,truth,observations,plane,initial


def test_multiview_recovers_known_extrinsics_and_predicts_unfitted_pose():
    K,prior,truth,observations,plane,initial=synthetic_problem();before=K.copy()
    # Pose order is deliberately shuffled, with last physical pose not fitted.
    training=[observations[i] for i in [4,0,6,2,1,5,3]]
    solved,diagnostics,_=fit_model(K,prior,training,plane,initial,regularize=False)
    assert solved.success
    assert diagnostics['observable']
    np.testing.assert_allclose(solved.x,truth,atol=2e-5)
    E,p=unpack(solved.x,prior)
    np.testing.assert_allclose(project(K,np.array(observations[7]['base_from_link6']),E,p),observations[7]['corners_uv'],atol=.01)
    np.testing.assert_array_equal(K,before)


def test_repeated_single_pose_reports_weak_observability():
    K,prior,truth,observations,plane,initial=synthetic_problem()
    solved,diagnostics,_=fit_model(K,prior,[observations[0]]*8,plane,initial,regularize=False)
    assert diagnostics['relative_rank_1e4']<10
    assert not diagnostics['observable']


def test_empty_metric_set_is_json_null_and_zero_count():
    import json
    from calibrate_playground_multiview import metric_errors
    K,prior,truth,_,_,_=synthetic_problem()
    result=metric_errors(truth,prior,K,[])
    assert result['n']==0 and result['n_points']==0
    assert result['point_rmse_px'] is None
    json.dumps(result,allow_nan=False)


def test_bad_calibration_input_is_rejected():
    import pytest
    K,prior,truth,observations,plane,initial=synthetic_problem()
    K[0,0]=np.nan
    with pytest.raises(ValueError,match='K must'):
        fit_model(K,prior,observations,plane,initial)


def test_cli_refuses_existing_evidence_directory(tmp_path,monkeypatch):
    import pytest
    from calibrate_playground_multiview import main
    marker=tmp_path/'keep.json';marker.write_text('{"keep": true}')
    monkeypatch.setattr(sys,'argv',['calibrate','--transfer','missing', '--output',str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code==2
    assert marker.read_text()=='{"keep": true}'
