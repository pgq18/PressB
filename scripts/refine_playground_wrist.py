#!/usr/bin/env python3
"""Offline wrist-only visual alignment with fixed K, geometry, cube poses and mount.

Never calls hardware or simulation. Works in link6 coordinates for fingers, using
measured real gripper widths. Static cube observations are restricted to precontact.
"""
import argparse, copy, hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from calibrate_playground_video import D,C,correction,project,load_observations,binary_stl_vertices

def horizontal_edges(uv, y):
    # Preserve STL faces: the convex hull incorrectly bridges concave finger body edges.
    triangles=uv.reshape(-1,3,2); crosses=[]
    for i in range(3):
        a=triangles[:,i];b=triangles[:,(i+1)%3]
        ok=((np.minimum(a[:,1],b[:,1])<=y)&(np.maximum(a[:,1],b[:,1])>=y)&(np.abs(a[:,1]-b[:,1])>1e-10))
        a=a[ok];b=b[ok]
        crosses.extend((a[:,0]+(b[:,0]-a[:,0])*(y-a[:,1])/(b[:,1]-a[:,1])).tolist())
    if not crosses:
        v=uv[np.argmin(abs(uv[:,1]-y))]
        return np.array([v[0],v[0]])
    return np.array([min(crosses),max(crosses)])

def finger_triangles(width,link):
    vertices=binary_stl_vertices(ROOT/f'vendor/robot_lab/source/robot_lab/data/Robots/Agilex/PIPER/meshes/{link}.STL')
    rpy=[1.5708,0,0] if link=='link7' else [1.5708,0,-3.1416]
    vertices=vertices+np.array([0,0,width/2])
    return vertices@Rotation.from_euler('xyz',rpy).as_matrix().T+np.array([0,0,.1358])

def validate_edges(observations,resolution):
    if not observations or not any(o.get('split')=='train' for o in observations):
        raise ValueError('Need nonempty finger observations with a training split')
    width,height=resolution
    for observation in observations:
        if observation.get('split') not in ['train','holdout']:
            raise ValueError('Observation split must be train or holdout')
        if observation.get('link','link8') not in ['link7','link8']:
            raise ValueError('Only bundled link7/link8 finger meshes are supported')
        opening=observation['width_m']
        if not np.isfinite(opening) or not 0<=opening<=.07:
            raise ValueError('Finger width must be finite and between 0 and 0.07 m')
        if 'tip_v' in observation:
            if not np.isfinite(observation['tip_v']) or not 0<=observation['tip_v']<height:
                raise ValueError('Tip v must lie inside the native image')
            if not np.isfinite(observation.get('tip_sigma_px',2)) or observation.get('tip_sigma_px',2)<=0:
                raise ValueError('Tip uncertainty must be finite and positive')
        if not observation.get('rows') and 'tip_v' not in observation:
            raise ValueError('Observation contains no finger measurements')
        for row in observation.get('rows',[]):
            values=[row['v'],row['left_u'],row['right_u'],row.get('sigma_px',3)]
            if not np.isfinite(values).all() or not 0<=row['v']<height:
                raise ValueError('Contour rows must be finite and inside the image')
            if not 0<=row['left_u']<=row['right_u']<width or row.get('sigma_px',3)<=0:
                raise ValueError('Contour edges must be ordered inside the image, with positive uncertainty')

def run(config,annotations,edges,output,cube_sigma=5.,prior_deg=.3,prior_mm=.5):
    for name,value in [('cube_sigma',cube_sigma),('prior_deg',prior_deg),('prior_mm',prior_mm)]:
        if not np.isfinite(value) or value<=0:
            raise ValueError(f'{name} must be finite and positive')
    output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(config.read_text()); obs,_=load_observations(annotations,cfg)
    # Legacy v04 annotations were manually checked before the first contact.
    # New captures must explicitly mark static_before_first_contact=True.
    legacy_static_names={'pregrasp_10s','pregrasp_20s','pregrasp_28s','pregrasp_34s'}
    obs=[o for o in obs if o['camera']=='wrist' and o.get('cube_corners') and
         (o.get('static_before_first_contact') is True or
          ('static_before_first_contact' not in o and o['name'] in legacy_static_names))]
    edge_doc=json.loads(edges.read_text()); eo=edge_doc['observations']
    validate_edges(eo,cfg['wrist_camera']['resolution'])
    cam=cfg['wrist_camera'];K=np.array([[cam['fx'],0,cam['cx']],[0,cam['fy'],cam['cy']],[0,0,1.]])
    prior=np.array(cam['link6_from_camera_usd'])@D
    cubes={o['name']:o for o in cfg['calibration_objects']}; points={}
    for name,c in cubes.items():
        points[name]=C*c['size'][0]@Rotation.from_euler('z',c['yaw_deg'],degrees=True).as_matrix().T+np.array([*c['center_xy'],-cfg['robot_base_height_above_table_m']+c['size'][0]])
    for o in eo:o['_verts']=finger_triangles(o['width_m'],o.get('link','link8'))
    def features(p,split=None):
        camera=correction(p,prior); result=[]
        for o in obs:
            if split and o['split']!=split:continue
            T=np.array(o['base_from_link6'])@camera
            for name,xy in o['cube_corners'].items():
                pred=project(K,T,points[name]);target=np.array(xy,float)
                result.append(dict(name=o['name'],kind='cube',object=name,split=o['split'],pred=pred,target=target,sigma=cube_sigma))
        for o in eo:
            if split and o['split']!=split:continue
            uv=project(K,camera,o['_verts'])
            if 'tip_v' in o:
                result.append(dict(name=o['name'],kind='finger_tip',width_m=o['width_m'],split=o['split'],pred=np.array([uv[:,1].min()]),target=np.array([o['tip_v']]),sigma=o.get('tip_sigma_px',2.)))
            for row in o.get('rows',[]):
                pred=horizontal_edges(uv,row['v']);target=np.array([row['left_u'],row['right_u']]);ok=np.isfinite(target)
                result.append(dict(name=o['name'],kind='finger_edges',width_m=o['width_m'],split=o['split'],v=row['v'],pred=pred[ok],target=target[ok],sigma=row.get('sigma_px',3.)))
        return result
    def data(p,split='train'):
        return np.concatenate([((f['pred']-f['target'])/f['sigma']).ravel() for f in features(p,split)])
    sigma=np.array([*[np.deg2rad(prior_deg)]*3,*[prior_mm*.001]*3]); bounds=np.array([*[np.deg2rad(3.)]*3,*[.008]*3])
    fun=lambda p:np.r_[data(p),p/sigma]
    res=least_squares(fun,np.zeros(6),bounds=(-bounds,bounds),x_scale='jac',loss='soft_l1',f_scale=2,max_nfev=600,ftol=1e-10,xtol=1e-10,gtol=1e-10)
    def metrics(p):
        out={}
        for split in ['train','holdout']:
            entries=[];groups={}
            for f in features(p,split):
                err=f['pred']-f['target'];group=f['kind'];groups.setdefault(group,[]).extend(err.ravel().tolist())
                a={k:v for k,v in f.items() if k not in ['pred','target']};a.update(predicted=f['pred'].tolist(),target=f['target'].tolist(),residual=err.tolist(),rmse_px=float(np.sqrt(np.mean(err**2))));entries.append(a)
            out[split]=dict(groups={k:dict(n=len(v),rmse_px=float(np.sqrt(np.mean(np.array(v)**2))),max_abs_px=float(np.max(np.abs(v)))) for k,v in groups.items()},features=entries)
        return out
    scale=np.array([*[np.deg2rad(1)]*3,*[.003]*3]);eps=1e-5
    J=np.column_stack([(data(res.x+np.eye(6)[i]*scale[i]*eps)-data(res.x-np.eye(6)[i]*scale[i]*eps))/(2*eps) for i in range(6)])
    U,s,V=np.linalg.svd(J,full_matrices=False)
    candidate=copy.deepcopy(cfg);candidate['wrist_camera']['link6_from_camera_usd']=(correction(res.x,prior)@D).tolist()
    candidate.setdefault('reconstruction',{})['wrist_extrinsics_sha256']=hashlib.sha256(
        json.dumps(candidate['wrist_camera']['link6_from_camera_usd']).encode()).hexdigest()
    report=dict(used_cube_observations=[dict(name=o['name'],split=o['split'],objects=list(o['cube_corners']),gate='explicit_precontact' if o.get('static_before_first_contact') is True else 'legacy_v04_verified_precontact_name') for o in obs],used_finger_observations=[dict(name=o['name'],split=o['split'],width_m=o['width_m']) for o in eo],success=bool(res.success),optimizer_message=res.message,parameters=res.x.tolist(),delta_rotation_degrees=np.rad2deg(res.x[:3]).tolist(),delta_translation_mm=(1000*res.x[3:]).tolist(),rotation_norm_deg=float(np.linalg.norm(np.rad2deg(res.x[:3]))),translation_norm_mm=float(np.linalg.norm(1000*res.x[3:])),before=metrics(np.zeros(6)),after=metrics(res.x),data_identifiability=dict(scales=scale.tolist(),singular_values=s.tolist(),condition_number=float(s[0]/max(s[-1],1e-12)),rank=int((s>s[0]*1e-4).sum()),weakest_scaled_direction=V[-1].tolist()),bounds=dict(rotation_each_deg=3,translation_each_mm=8),regularization=dict(rotation_each_deg=prior_deg,translation_each_mm=prior_mm,cube_sigma_px=cube_sigma),fixed=['real intrinsics','mount height','global camera','robot URDF/joint feedback','cube dimensions and poses','actual finger width'],inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [config,annotations,edges,Path(__file__)]},limitations=['Manual contour edges are approximate; adjacent rows and repeated widths are correlated.','Finger STL surface shape and real finger manufacturing/rubber pads may differ.','Optimizer-excluded widths and frames were reviewed for model selection; they are development validation, not a blind test.','Only precontact cube top-face corners are constrained.','Width disagreement between real contact and physics is not attributed to camera.'])
    (output/'candidate_config.json').write_text(json.dumps(candidate,indent=2)+'\n');(output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['success','delta_rotation_degrees','delta_translation_mm','data_identifiability']}));return candidate,report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/calibration/playground_20261107_before_wrist_refine.json');p.add_argument('--annotations',type=Path,default=ROOT/'configs/calibration/playground_20261107_retry08_video_annotations_v04.json');p.add_argument('--edges',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cube-sigma',type=float,default=5);p.add_argument('--prior-deg',type=float,default=.3);p.add_argument('--prior-mm',type=float,default=.5);a=p.parse_args();run(a.config,a.annotations,a.edges,a.output,a.cube_sigma,a.prior_deg,a.prior_mm)
