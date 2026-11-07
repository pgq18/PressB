#!/usr/bin/env python3
"""Offline image/FK registration. Never communicates with robot/simulator.

Live intrinsics, URDF, cube sizes, measured joints and recorded commands are
immutable. Free variables: two SE(3) camera corrections and static cube XY/yaw.
Manual endpoint observations constrain projected STL fingertip silhouettes,
not falsely identified exact mesh vertices. Optional global bare-table depth is
a fit constraint; wrist depth remains an independent diagnostic.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pressb.robots.piper.kinematics import PiperKinematics
D=np.diag([1.,-1.,-1.,1.]); C=np.array([[-.5,-.5,0],[-.5,.5,0],[.5,.5,0],[.5,-.5,0]])

def correction(p,T):
    result=T.copy();result[:3,:3]=Rotation.from_rotvec(p[:3]).as_matrix()@T[:3,:3];result[:3,3]+=p[3:6];return result

def project(K,T,points):
    v=(np.asarray(points)-T[:3,3])@T[:3,:3]
    if np.any(v[:,2]<=.01): raise ValueError('Point behind/at camera')
    uv=v@K.T;return uv[:,:2]/uv[:,2:]

def binary_stl_vertices(path):
    """Read the bundled binary STL without optional mesh-processing packages."""
    raw=Path(path).read_bytes()
    if len(raw)<84: raise ValueError('Truncated binary STL header')
    count=int.from_bytes(raw[80:84],'little')
    if count<=0 or len(raw)!=84+50*count: raise ValueError('Invalid binary STL face count or size')
    dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')])
    vertices=np.frombuffer(raw,dtype=dtype,count=count,offset=84)['vertices'].reshape(-1,3).astype(float)
    if not np.isfinite(vertices).all(): raise ValueError('Nonfinite binary STL vertex')
    return vertices

def make_fingers(width):
    result={}
    for name,rpy in [('link7',[1.5708,0,0]),('link8',[1.5708,0,-3.1416])]:
        v=np.unique(binary_stl_vertices(ROOT/f'vendor/robot_lab/source/robot_lab/data/Robots/Agilex/PIPER/meshes/{name}.STL'),axis=0)
        v=np.unique(np.round(v,7),axis=0)
        # Both URDF fingers slide +half-width along their own local +Z.
        v=v+np.array([0,0,width/2]);v=v@Rotation.from_euler('xyz',rpy).as_matrix().T+np.array([0,0,.1358])
        result[name]=v
    return result

def load_observations(path,cfg):
    doc=json.loads(path.read_text());obs=doc['observations']; kin=PiperKinematics(ROOT/cfg['robot_urdf'],tip_offset=0)
    for o in obs:
        o['base_from_link6']=kin.fk(np.array(o['q_rad_m'][:6])).tolist()
        o['_fingers']=make_fingers(o['q_rad_m'][6])
    return obs,doc

def global_plane_residual(camera_cv, plane, table_z):
    """Plane n_cv·p_cv=d maps to normal_base·p_base=offset (up normal)."""
    T=np.asarray(camera_cv,float);n=-T[:3,:3]@np.array(plane['raw_normal_cv'])
    offset=float(n@T[:3,3]-plane['raw_plane_distance_m'])
    return np.r_[n[:2]/np.deg2rad(.15),(offset-n[2]*table_z)/.0015]

def solve(cfg,observations,*,finger_weight=1,prior_scale=1,global_plane=None):
    if not np.isfinite(finger_weight) or finger_weight<=0 or not np.isfinite(prior_scale) or prior_scale<=0:
        raise ValueError('finger_weight and prior_scale must be finite and positive')
    priors={'global':np.array(cfg['up_camera']['base_from_camera_usd'])@D,'wrist':np.array(cfg['wrist_camera']['link6_from_camera_usd'])@D}
    intr={}
    for name,key in [('global','up_camera'),('wrist','wrist_camera')]:
        c=cfg[key];intr[name]=np.array([[c['fx'],0,c['cx']],[0,c['fy'],c['cy']],[0,0,1.]])
    cubes={x['name']:x for x in cfg['calibration_objects']};names=list(cubes)
    init=np.concatenate([np.zeros(12)]+[np.array([*cubes[n]['center_xy'],np.deg2rad(cubes[n]['yaw_deg'])]) for n in names])
    table_z=-cfg['robot_base_height_above_table_m']
    def unpack(p):
        cameras={'global':correction(p[:6],priors['global']),'wrist':correction(p[6:12],priors['wrist'])}
        points={}
        for i,n in enumerate(names):
            edge=cubes[n]['size'][0];v=p[12+3*i:15+3*i];points[n]=C*edge@Rotation.from_euler('z',v[2]).as_matrix().T+np.array([*v[:2],table_z+edge])
        return cameras,points
    def features(p,o):
        cameras,points=unpack(p);name=o['camera'];T=cameras[name]
        if name=='wrist': T=np.array(o['base_from_link6'])@T
        result=[]
        for n,uv in o.get('cube_corners',{}).items():
            pred=project(intr[name],T,points[n]); target=np.array([[np.nan,np.nan] if x is None else x for x in uv],float);ok=np.isfinite(target).all(1)
            if ok.any():result.append(dict(kind='cube',object=n,predicted=pred[ok],target=target[ok],sigma=o.get('cube_sigma_px',2.)))
        for a in o.get('finger_silhouettes',[]):
            v=o['_fingers'][a['link']];base=np.array(o['base_from_link6']);v=v@base[:3,:3].T+base[:3,3];uv=project(intr[name],T,v)
            axis={'u':0,'v':1}[a['axis']];idx=np.argmin(uv[:,axis]) if a['extremum']=='min' else np.argmax(uv[:,axis]);pred=uv[idx,axis]
            if 'uv' in a:
                hull=uv[ConvexHull(uv).vertices];sign=-1 if a['extremum']=='min' else 1
                weight=np.exp(np.clip(sign*(hull[:,axis]-pred)/a.get('band_px',.75),-700,0));p2=(hull*weight[:,None]).sum(0)/weight.sum();p2[axis]=pred
                result.append(dict(kind='finger',object=a['link'],predicted=p2,target=np.array(a['uv']),sigma=a.get('sigma_px',3.)/finger_weight))
            else:
                result.append(dict(kind='finger',object=a['link'],predicted=np.array([pred]),target=np.array([a['value_px']]),sigma=a.get('sigma_px',3.)/finger_weight))
        return result
    train=[o for o in observations if o.get('split','train')=='train']
    def data(p):
        terms=[((f['predicted']-f['target'])/f['sigma']).ravel() for o in train for f in features(p,o)]
        if global_plane is not None:
            cameras,_=unpack(p);terms.append(global_plane_residual(cameras['global'],global_plane,table_z))
        return np.concatenate(terms)
    sigma=np.concatenate([np.full(3,np.deg2rad(4)),np.full(3,.025),np.full(3,np.deg2rad(5)),np.full(3,.025)]+[np.array([.03,.03,np.deg2rad(30)]) for n in names])*prior_scale
    def fun(p):return np.r_[data(p),(p-init)/sigma]
    bounds=np.concatenate([np.full(3,np.deg2rad(12)),np.full(3,.06),np.full(3,np.deg2rad(12)),np.full(3,.06)]+[np.array([.08,.08,np.deg2rad(65)]) for n in names])
    solution=least_squares(fun,init,bounds=(init-bounds,init+bounds),x_scale='jac',loss='soft_l1',f_scale=2,max_nfev=1200,ftol=1e-10,xtol=1e-10,gtol=1e-10)
    def metrics(p,split):
        items=[];errs={}
        for o in observations:
            if split and o.get('split','train')!=split:continue
            for f in features(p,o):
                e=f['predicted']-f['target'];k=f"{o['camera']}_{f['kind']}";errs.setdefault(k,[]).extend(e.ravel().tolist());items.append(dict(name=o['name'],camera=o['camera'],kind=f['kind'],object=f['object'],coordinate_rmse_px=float(np.sqrt(np.mean(e*e))),predicted=f['predicted'].tolist(),target=f['target'].tolist()))
        return dict(groups={k:dict(n_coordinates=len(v),coordinate_rmse_px=float(np.sqrt(np.mean(np.square(v))))) for k,v in errs.items()},features=items)
    scales=np.concatenate([np.full(3,np.deg2rad(3)),np.full(3,.01),np.full(3,np.deg2rad(3)),np.full(3,.01)]+[np.array([.01,.01,np.deg2rad(10)]) for n in names])
    step=1e-5; J=np.column_stack([(data(solution.x+np.eye(len(init))[i]*scales[i]*step)-data(solution.x-np.eye(len(init))[i]*scales[i]*step))/(2*step) for i in range(len(init))]);_,s,vh=np.linalg.svd(J,full_matrices=False)
    report=dict(success=bool(solution.success),optimizer_message=solution.message,parameters=solution.x.tolist(),parameter_delta=(solution.x-init).tolist(),camera_delta={n:dict(rotation_deg=np.rad2deg(solution.x[i:i+3]).tolist(),translation_mm=(solution.x[i+3:i+6]*1000).tolist()) for n,i in [('global',0),('wrist',6)]},before={split:metrics(init,split) for split in ['train','holdout']},after={split:metrics(solution.x,split) for split in ['train','holdout']},identifiability=dict(data_scaled_singular_values=s.tolist(),relative_rank_1e4=int((s>s[0]*1e-4).sum()),parameter_count=len(init),condition_number=float(s[0]/max(s[-1],1e-12)),weakest_scaled_direction=vh[-1].tolist(),parameter_scales=scales.tolist()),bound_fraction=(np.abs(solution.x-init)/bounds).tolist())
    candidate=copy.deepcopy(cfg);cams,pts=unpack(solution.x)
    candidate['up_camera']['base_from_camera_usd']=(cams['global']@D).tolist();candidate['up_camera']['eye_base_m']=cams['global'][:3,3].tolist();candidate['up_camera']['target_base_m']=(cams['global'][:3,3]+cams['global'][:3,2]*.5).tolist();candidate['wrist_camera']['link6_from_camera_usd']=(cams['wrist']@D).tolist()
    for i,obj in enumerate(candidate['calibration_objects']):obj['center_xy']=solution.x[12+3*i:14+3*i].tolist();obj['yaw_deg']=float(np.rad2deg(solution.x[14+3*i]))
    candidate.setdefault('reconstruction',{})['wrist_extrinsics_sha256']=hashlib.sha256(json.dumps(candidate['wrist_camera']['link6_from_camera_usd']).encode()).hexdigest()
    candidate['reconstruction']['video_registration_candidate']={'status':'Offline visual-registration candidate, not absolute physical calibration','fixed_intrinsics':True,'fixed_robot_kinematics':True,'fixed_cube_sizes':True,'fixed_base_height_m':cfg['robot_base_height_above_table_m'],'global_depth_plane_used':global_plane is not None,'actual_robot_or_commands_modified':False}
    return candidate,report

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--finger-weight',type=float,default=1);p.add_argument('--prior-scale',type=float,default=1);p.add_argument('--depth-plane-report',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(a.config.read_text());obs,doc=load_observations(a.annotations,cfg);plane=json.loads(a.depth_plane_report.read_text())['results']['home']['global'] if a.depth_plane_report else None;candidate,report=solve(cfg,obs,finger_weight=a.finger_weight,prior_scale=a.prior_scale,global_plane=plane)
    report.update(input_asset_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [ROOT/cfg['robot_urdf'],*[ROOT/f'vendor/robot_lab/source/robot_lab/data/Robots/Agilex/PIPER/meshes/{n}.STL' for n in ['link7','link8']]]},depth_input_sha256=None if a.depth_plane_report is None else hashlib.sha256(a.depth_plane_report.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),parameter_names=[f'{camera}_{p}' for camera in ['global','wrist'] for p in ['rotvec_x_rad','rotvec_y_rad','rotvec_z_rad','tx_m','ty_m','tz_m']]+[f"{c['name']}_{p}" for c in cfg['calibration_objects'] for p in ['x_m','y_m','yaw_rad']],global_plane_constraint=None if plane is None else dict(source=str(a.depth_plane_report),normal_sigma_deg=.15,offset_sigma_m=.0015,scope='Far-field global bare-table plane only; wrist depth remains diagnostic'),source_config=str(a.config),source_annotations=str(a.annotations),config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),annotations_sha256=hashlib.sha256(a.annotations.read_bytes()).hexdigest(),fixed=['Live camera K','URDF and joint zeros','Measured q and gripper opening','Recorded command trajectory','Cube sizes: '+json.dumps({c['name']:c['size'] for c in cfg['calibration_objects']}),'Base height: '+str(cfg['robot_base_height_above_table_m'])+'m'],limitations=['Manual silhouette correspondence is approximate and its error is represented by pixel sigma.','One task trajectory cannot identify all camera/kinematic systematic errors.','This is visual registration, not absolute metrology or proof of changed physical contact.','Fitted static cube coordinates apply only before first contact.','The optimizer-excluded validation images were inspected during method development; these are not a blind independent test set.','Repeated unchanged global cube landmarks and repeated same-opening wrist finger supports are correlated; held-out camera time alone does not make these independent spatial calibration data.','Global home bare-table depth is a fit constraint when global_plane_constraint is present; global retreat capture is a repeat-capture check. Wrist home/retreat depth is never fitted and remains an independent diagnostic.'])
    (a.output/'candidate_config.json').write_text(json.dumps(candidate,indent=2)+'\n');(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['success','camera_delta','identifiability']},indent=2))
if __name__=='__main__':main()
