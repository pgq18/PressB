"""Offline registration must preserve metrology inputs and exclude holdouts."""
from __future__ import annotations
import copy
import importlib.util
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/calibrate_playground_video.py'
spec = importlib.util.spec_from_file_location('playground_video_calibration', SCRIPT)
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)


def synthetic():
    up = np.eye(4); up[:3,3] = [.3,0,1]
    wrist = np.eye(4); wrist[:3,3] = [0,0,.5]
    intr = dict(fx=600.,fy=601.,cx=320.,cy=240.,resolution=[640,480])
    cfg = dict(up_camera={**intr,'base_from_camera_usd':up.tolist()},
        wrist_camera={**intr,'link6_from_camera_usd':wrist.tolist()},
        robot_base_height_above_table_m=.014,
        calibration_objects=[dict(name='GreenCube',size=[.029]*3,center_xy=[.32,.08],yaw_deg=21.),
                             dict(name='BlueCube',size=[.030]*3,center_xy=[.33,.17],yaw_deg=-4.)],
        objects=[],robot_urdf='unchanged.urdf')
    K=np.array([[600.,0,320.],[0,601.,240.],[0,0,1.]])
    observations=[]
    for i in range(4):
        base=np.eye(4);base[:3,3]=[.05*i,.01*i,.01*i]
        base[:3,:3]=Rotation.from_euler('xyz',[i*.03,i*.02,-i*.04]).as_matrix()
        for name in ['global','wrist']:
            T=up@cal.D if name=='global' else base@wrist@cal.D
            cubes={}
            for c in cfg['calibration_objects']:
                e=c['size'][0];points=cal.C*e@Rotation.from_euler('z',c['yaw_deg'],degrees=True).as_matrix().T+[*c['center_xy'],e-.014]
                cubes[c['name']]=cal.project(K,T,points).tolist()
            observations.append(dict(name=f'pose{i}',camera=name,split='holdout' if i==3 else 'train',
                base_from_link6=base.tolist(),cube_corners=cubes,_fingers={}))
    return cfg,observations


def test_holdout_observations_never_change_fit():
    cfg,obs=synthetic();a,ra=cal.solve(cfg,obs)
    corrupted=copy.deepcopy(obs)
    for o in corrupted:
        if o['split']=='holdout':
            for key,uv in o['cube_corners'].items():o['cube_corners'][key]=(np.array(uv)+[80,-40]).tolist()
    b,rb=cal.solve(cfg,corrupted)
    np.testing.assert_allclose(ra['parameters'],rb['parameters'],atol=1e-12)
    assert rb['after']['holdout']['groups']['wrist_cube']['coordinate_rmse_px']>50
    assert ra['after']['holdout']['groups']['wrist_cube']['coordinate_rmse_px']<1e-6


def test_exact_registration_preserves_fixed_inputs_and_proper_rotations():
    cfg,obs=synthetic();original=copy.deepcopy(cfg);candidate,report=cal.solve(cfg,obs)
    assert cfg==original
    for key in ['robot_urdf','objects','robot_base_height_above_table_m']:assert candidate[key]==cfg[key]
    for key,tf in [('up_camera','base_from_camera_usd'),('wrist_camera','link6_from_camera_usd')]:
        for k in ['fx','fy','cx','cy','resolution']:assert candidate[key][k]==cfg[key][k]
        R=np.array(candidate[key][tf])[:3,:3];np.testing.assert_allclose(R.T@R,np.eye(3),atol=1e-12);assert np.linalg.det(R)>0.999999
    for a,b in zip(candidate['calibration_objects'],cfg['calibration_objects']):assert a['size']==b['size']
    assert max(report['bound_fraction'])<1e-7
    import hashlib,json
    assert candidate['reconstruction']['wrist_extrinsics_sha256']==hashlib.sha256(json.dumps(candidate['wrist_camera']['link6_from_camera_usd']).encode()).hexdigest()


def test_projection_rejects_behind_camera_points():
    import pytest
    with pytest.raises(ValueError,match='behind'):
        cal.project(np.eye(3),np.eye(4),[[0,0,-1]])


def test_global_depth_plane_sign_and_units():
    T=np.diag([1.,-1.,-1.,1.]);T[2,3]=1.
    plane=dict(raw_normal_cv=[0,0,1.],raw_plane_distance_m=1.014)
    np.testing.assert_allclose(cal.global_plane_residual(T,plane,-.014),[0,0,0],atol=1e-10)
    T[2,3]+=.01
    np.testing.assert_allclose(cal.global_plane_residual(T,plane,-.014),[0,0,.01/.0015],atol=1e-10)


def test_binary_stl_rejects_truncation_and_preserves_vertices(tmp_path):
    import struct,pytest
    path=tmp_path/'one.stl';values=[0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,1.,0.]
    path.write_bytes(bytes(80)+struct.pack('<I',1)+struct.pack('<12fH',*values,0))
    np.testing.assert_array_equal(cal.binary_stl_vertices(path),[[0,0,0],[1,0,0],[0,1,0]])
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError,match='face count or size'):cal.binary_stl_vertices(path)


def test_versioned_input_loads_without_optional_mesh_packages():
    import json
    root=SCRIPT.parents[1]
    cfg=json.loads((root/'configs/calibration/playground_20261107_retry08_video_baseline.json').read_text())
    obs,_=cal.load_observations(root/'configs/calibration/playground_20261107_retry08_video_annotations.json',cfg)
    assert obs and all(set(o['_fingers'])=={'link7','link8'} for o in obs)
    assert all(np.isfinite(o['_fingers']['link7']).all() for o in obs)


def test_cli_refuses_existing_output_before_reading_inputs(tmp_path,monkeypatch):
    import pytest,sys
    marker=tmp_path/'preserve';marker.write_text('original')
    monkeypatch.setattr(sys,'argv',[str(SCRIPT),'--config','missing.json','--annotations','missing.json','--output',str(tmp_path)])
    with pytest.raises(FileExistsError):cal.main()
    assert marker.read_text()=='original'
