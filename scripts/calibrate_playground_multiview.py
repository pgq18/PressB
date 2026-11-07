#!/usr/bin/env python3
"""Offline multi-pose camera registration; never commands a robot.

Fit constant wrist camera-to-link6 extrinsics and a static cube from pre-contact
images, live fixed RGB intrinsics and a reliable initial RGB-D table plane.
Output is a review-only candidate, not an absolute hardware calibration.
The base/table zero-height convention, URDF joint zeros and 30mm cube prior are
explicit assumptions. Temporal holdouts are not used in the optimization.
This runner currently consumes the 2026-11-07 annotated green-cube sequence,
including its initial cube pixels and 59.4mm stationary gripper-opening check.
Other sessions require their own annotations and measured gripper width.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import hashlib
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pressb.robots.piper.kinematics import PiperKinematics
from fit_playground_camera import fit_plane,ray

D=np.diag([1.,-1.,-1.,1.])
CORNERS=np.array([[-.5,-.5,0],[-.5,.5,0],[.5,.5,0],[.5,-.5,0]])


def project(K, base_from_link, extrinsic_cv, points):
    camera=base_from_link@extrinsic_cv
    local=(camera[:3,:3].T@(points-camera[:3,3]).T).T
    uv=local@K.T
    return uv[:,:2]/uv[:,2:]


def unpack(parameters, prior):
    e=prior.copy()
    e[:3,:3]=Rotation.from_rotvec(parameters[:3]).as_matrix()@prior[:3,:3]
    e[:3,3]=prior[:3,3]+parameters[3:6]
    xy=parameters[6:8];yaw=parameters[8];edge=parameters[9]
    center=np.array([*xy,edge+(parameters[10] if len(parameters)>10 else 0.)])
    points=center+(Rotation.from_euler('z',yaw).as_matrix()@(CORNERS*edge).T).T
    return e,points


def metric_errors(parameters, prior, K, observations):
    """Euclidean corner error; missing/out-of-view corners are explicitly masked."""
    e,p=unpack(parameters,prior)
    results=[]; all_errors=[]
    for o in observations:
        predicted=project(K,np.array(o['base_from_link6']),e,p)
        mask=np.asarray(o.get('valid_mask',[True]*4),dtype=bool)
        target=np.asarray(o['corners_uv'],dtype=float)
        if mask.shape!=(4,) or target.shape!=(4,2):
            raise ValueError('Metric observations require four corner slots and a four-entry valid mask')
        if not np.all(np.isfinite(target[mask])):
            raise ValueError('Visible corner coordinates must be finite')
        if not np.any(mask):
            continue
        error=(predicted-target)[mask];all_errors.append(error)
        results.append(dict(name=o['name'],elapsed_s=o.get('elapsed_s'),n_points=int(mask.sum()),
            coordinate_rmse_px=float(np.sqrt(np.mean(error**2))),
            point_rmse_px=float(np.sqrt(np.mean(np.sum(error**2,axis=1)))),
            max_point_error_px=float(np.linalg.norm(error,axis=1).max()),
            valid_mask=mask.tolist(),projected_uv=predicted.tolist()))
    if not all_errors:
        return dict(n=0,n_points=0,point_rmse_px=None,coordinate_rmse_px=None,frames=[])
    errors=np.concatenate(all_errors,axis=0)
    return dict(n=len(results),n_points=len(errors),
                point_rmse_px=float(np.sqrt(np.mean(np.sum(errors**2,axis=1)))),
                coordinate_rmse_px=float(np.sqrt(np.mean(errors**2))),frames=results)


def fit_model(K, prior, observations, plane, initial, *, regularize=True, finger=None):
    """Return bounded fit with data-only Jacobian diagnostics.

    Parameter units: rotvec rad3, translation m3, cubeXY m2, yaw rad, edge m.
    Pixel sigma2px, table normal0.005rad and offset2mm, depth cube center2mm.
    Weak hand-eye directions are softly held near the supplied prior; no joint
    zeros, timestamp offsets, camera intrinsics or physical joint limits change.
    """
    initial=np.array(initial,float)
    K=np.asarray(K,float);prior=np.asarray(prior,float)
    if K.shape!=(3,3) or not np.all(np.isfinite(K)) or K[0,0]<=0 or K[1,1]<=0:
        raise ValueError('K must be a finite3x3 matrix with positive focal lengths')
    if prior.shape!=(4,4) or not np.all(np.isfinite(prior)):
        raise ValueError('Camera prior must be a finite4x4 transform')
    if initial.shape not in [(10,),(11,)] or not np.all(np.isfinite(initial)):
        raise ValueError('Initial parameters must contain10 or11 finite values')
    if not observations:
        raise ValueError('At least one training observation is required')
    for obs in observations:
        uv=np.asarray(obs['corners_uv'],float);base=np.asarray(obs['base_from_link6'],float)
        if uv.shape!=(4,2) or not np.all(np.isfinite(uv)):
            raise ValueError('Training observations require four finite2D corner pairs')
        if base.shape!=(4,4) or not np.all(np.isfinite(base)):
            raise ValueError('Observed flange pose must be a finite4x4 transform')
    def data_residual(x):
        e,p=unpack(x,prior)
        terms=[((project(K,np.array(o['base_from_link6']),e,p)-o['corners_uv'])/2).ravel()
               for o in observations]
        if plane is not None:
            T=np.array(plane['base_from_link6'])@e;n=np.array(plane['normal_cv']);d=plane['distance_m']
            normal=T[:3,:3]@(-n)
            # Plane z=0 is the scene/base convention, not independent metrology.
            terms.append(normal[:2]/.005)
            table_z=x[10] if len(x)>10 else 0.
            terms.append(np.array([(normal@T[:3,3]-d-normal[2]*table_z)/.002]))
            center_cv=T[:3,:3].T@(p.mean(0)-T[:3,3])
            terms.append((center_cv-np.array(plane['cube_center_cv_m']))/.002)
            terms.append(np.array([(x[9]-plane['cube_edge_m'])/.0015]))
        if finger is not None:
            terms.append(finger_residual(K,e,finger)/(2. if finger.get('mode')=='tip' else 4.))
        return np.concatenate(terms)
    def residual(x):
        value=data_residual(x)
        if regularize:
            # Conservative regularization for directions a single path observes weakly.
            value=np.r_[value,x[:3]/np.deg2rad(8.),x[3:6]/.025]
            if len(x)>10:value=np.r_[value,x[10]/.02]
        return value
    lower=np.r_[np.full(3,-np.deg2rad(15)),np.full(3,-.05),initial[6:8]-.06,-np.pi,.026]
    upper=np.r_[np.full(3,np.deg2rad(15)),np.full(3,.05),initial[6:8]+.06,np.pi,.034]
    if len(initial)>10:
        lower=np.r_[lower,-.03];upper=np.r_[upper,.03]
    solved=least_squares(residual,initial,bounds=(lower,upper),loss='soft_l1',f_scale=2.,
                         x_scale='jac',max_nfev=2000,ftol=1e-11,xtol=1e-11,gtol=1e-11)
    # Dimensionless practical scales:5deg,10mm,10mm,5deg,1mm.
    scales=np.r_[np.full(3,np.deg2rad(5)),np.full(3,.01),.01,.01,np.deg2rad(5),.001]
    if len(initial)>10:scales=np.r_[scales,.01]
    step=1e-5
    jac=np.column_stack([(data_residual(solved.x+np.eye(len(initial))[i]*scales[i]*step)-
                          data_residual(solved.x-np.eye(len(initial))[i]*scales[i]*step))/(2*step)
                         for i in range(len(initial))])
    _,s,vh=np.linalg.svd(jac,full_matrices=False)
    rank=int(np.sum(s>s[0]*1e-4))
    diagnostics=dict(data_only_scaled_singular_values=s.tolist(),relative_rank_1e4=rank,
         condition_number=float(s[0]/max(s[-1],1e-14)),weakest_scaled_direction=vh[-1].tolist(),
         practical_parameter_scales=scales.tolist(),
         observable=bool(rank==len(initial) and s[-1]>.1),
         caveat='Numerical local identifiability under fixed URDF/table assumptions; not absolute calibration confidence.')
    return solved,diagnostics,data_residual


def interpolate_q(rows,stamp):
    times=np.array([r['unix_s'] for r in rows])
    if len(times)<2 or not np.all(np.isfinite(times)) or np.any(np.diff(times)<=0):
        raise ValueError('Feedback requires at least two strictly increasing finite timestamps')
    if not np.isfinite(stamp):
        raise ValueError('Exposure timestamp must be finite')
    i=int(np.searchsorted(times,stamp))
    if i<=0 or i>=len(rows):
        raise ValueError('Camera exposure is outside recorded encoder feedback')
    a,b=rows[i-1],rows[i];span=b['unix_s']-a['unix_s'];alpha=(stamp-a['unix_s'])/span
    q=(1-alpha)*np.array(a['observed'][:6])+alpha*np.array(b['observed'][:6])
    return q,span


def make_plane(capture,cfg,kin):
    import cv2
    meta=json.loads((capture/'capture.json').read_text());intr=meta['cameras']['wrist']['color']
    p=fit_plane(capture/'wrist_depth.npz',intr,'wrist');n=np.array(p['normal']);d=p['distance_m']
    corners=np.array([[447,220],[401,226],[388,187],[433,181]],float)
    depth=np.load(capture/'wrist_depth.npz')['depth_m'];vv,uu=np.indices(depth.shape)
    xyz=np.stack([(uu-intr['cx'])/intr['fx']*depth,(vv-intr['cy'])/intr['fy']*depth,depth],-1)
    mask=np.zeros(depth.shape,np.uint8);cv2.fillConvexPoly(mask,corners.astype(np.int32),1)
    mask=cv2.erode(mask,np.ones((9,9),np.uint8)).astype(bool)
    valid=mask&(depth>.1)&(depth<1)
    top_d=float(np.median(xyz[valid]@n));top_points=np.array([ray(intr,uv)*top_d/(n@ray(intr,uv)) for uv in corners])
    lengths=np.linalg.norm(np.roll(top_points,-1,axis=0)-top_points,axis=1)
    # Height and horizontal edges both estimate an approximately cubic block.
    edge=float(np.mean(np.r_[lengths,d-top_d]))
    return dict(**p,normal_cv=p['normal'],base_from_link6=kin.fk(np.deg2rad(meta['feedback']['median_joints_deg'])).tolist(),
         cube_center_cv_m=top_points.mean(0).tolist(),cube_edge_m=edge,
         cube_height_depth_m=d-top_d,cube_horizontal_edges_depth_m=lengths.tolist(),
         cube_pixel_corners=corners.tolist(),capture=str(capture))


def plane_error(parameters,prior,plane):
    e,p=unpack(parameters,prior);T=np.array(plane['base_from_link6'])@e;n=np.array(plane['normal_cv']);normal=T[:3,:3]@(-n)
    closest=T[:3,:3]@(n*plane['distance_m'])+T[:3,3]
    return dict(normal_error_deg=float(np.degrees(np.arccos(np.clip(normal[2],-1,1)))),
                signed_plane_offset_m=float(normal@T[:3,3]-plane['distance_m']-normal[2]*(parameters[10] if len(parameters)>10 else 0.)),
                closest_point_base_m=closest.tolist(),constraint_role='fit constraint, not independent depth validation')


def make_finger_constraint(transfer,prior,K,mode="tip"):
    import cv2,trimesh
    from PIL import Image
    mesh=trimesh.load(ROOT/'vendor/robot_lab/source/robot_lab/data/Robots/Agilex/PIPER/meshes/link8.STL',process=False)
    T=np.eye(4);T[:3,:3]=Rotation.from_euler('xyz',[1.5708,0,-3.1416]).as_matrix();T[:3,3]=[0,0,.1358]
    slide=np.eye(4);slide[2,3]=.0594/2;T=T@slide
    vertices=(T@np.c_[mesh.vertices,np.ones(len(mesh.vertices))].T).T[:,:3]
    path=transfer/'edge_contact_analysis/wrist_holdout_t57.jpg';rgb=np.array(Image.open(path));roi=np.zeros(rgb.shape[:2],bool);roi[330:,270:470]=True
    mask=((rgb[:,:,1]<72)&(rgb[:,:,2]<95)&roi).astype(np.uint8)
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask);mask=labels==(1+np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    ys,xs=np.nonzero(mask);rows=[360,380,400,430]
    target=[float(np.median(xs[ys==ys.min()])),float(ys.min())]
    for row in rows:
        xx=np.where(mask[row])[0];target.extend([float(xx.min()),float(xx.max())])
    if mode=='tip':
        cp=(prior[:3,:3].T@(vertices-prior[:3,3]).T).T
        uv=cp@K.T;uv=uv[:,:2]/uv[:,2:]
        visible=cp[:,2]>.01
        low=uv[visible,1].min();selected=visible&(uv[:,1]<low+.1)
        return dict(tip_link6=vertices[selected].mean(0),target=np.array(target[:2]),reference=str(path),mode=mode)
    return dict(vertices_link6=vertices,rows=rows,target=np.array(target),reference=str(path),mode=mode)


def finger_residual(K,extrinsic,finger):
    from scipy.spatial import ConvexHull
    if finger.get('mode')=='tip':
        p=extrinsic[:3,:3].T@(finger['tip_link6']-extrinsic[:3,3]);uv=K@p
        return uv[:2]/uv[2]-finger['target']
    p=(extrinsic[:3,:3].T@(finger['vertices_link6']-extrinsic[:3,3]).T).T
    p=p[p[:,2]>.01];uv=p@K.T;uv=uv[:,:2]/uv[:,2:]
    hull=uv[ConvexHull(uv).vertices]
    minv=hull[:,1].min();top=hull[np.abs(hull[:,1]-minv)<.5]
    out=[float(top[:,0].mean()),float(minv)]
    for row in finger['rows']:
        xs=[]
        for a,b in zip(hull,np.roll(hull,-1,axis=0)):
            if min(a[1],b[1])<=row<=max(a[1],b[1]) and abs(b[1]-a[1])>1e-9:
                xs.append(a[0]+(row-a[1])*(b[0]-a[0])/(b[1]-a[1]))
        out.extend([min(xs),max(xs)] if xs else [1000.,1000.])
    return np.array(out)-finger['target']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transfer',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/piper_playground.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--finger-constraint',action='store_true',default=True,
                        help='Include finger geometry by default')
    parser.add_argument('--no-finger-constraint',dest='finger_constraint',action='store_false',
                        help='Diagnostic only: may expose weak camera/cube ambiguity')
    parser.add_argument('--fit-table-height',action='store_true')
    parser.add_argument('--finger-mode',choices=('tip','silhouette'),default='silhouette')
    args=parser.parse_args()
    if args.output.exists():
        parser.error(f'Output already exists; refusing to overwrite evidence: {args.output}')
    if not args.transfer.is_dir() or not args.config.is_file():
        parser.error('Input transfer directory and configuration file must exist')
    args.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(args.config.read_text());kin=PiperKinematics(ROOT/cfg['robot_urdf'],tip_offset=0)
    prior=np.array(cfg['wrist_camera']['link6_from_camera_usd'])@D;wi=cfg['wrist_camera']
    K=np.array([[wi['fx'],0,wi['cx']],[0,wi['fy'],wi['cy']],[0,0,1.]])
    rows=[json.loads(l) for l in (args.transfer/'playground_pregrasp_run_01/replay.jsonl').read_text().splitlines()]
    anns=json.loads((args.transfer/'calibration_landmarks_independent/landmarks.json').read_text())['annotations']
    observations=[];maxgap=0.
    for a in anns:
        q,gap=interpolate_q(rows,a['wrist_frame_unix_s']);maxgap=max(maxgap,gap)
        observations.append(dict(name=f"t{a['target_elapsed_s']}",elapsed_s=a['target_elapsed_s'],
            exposure_unix_s=a['wrist_frame_unix_s'],q_rad=q.tolist(),base_from_link6=kin.fk(q).tolist(),
            corners_uv=np.array(a['top_corners_px'])[[2,3,0,1]].tolist(),image=a['image'],
            split='heldout_contiguous' if 20<=a['target_elapsed_s']<=36 else 'train'))
    # A temporally contiguous middle block is held out before optimization.
    training=[o for o in observations if o['split']=='train'];holdout=[o for o in observations if o['split']!='train']
    independent=[]
    independent_file=args.transfer/'edge_contact_analysis/independent_holdout_corners.json'
    if independent_file.exists():
        for a in json.loads(independent_file.read_text())['entries']:
            q,gap=interpolate_q(rows,a['exposure_unix_s']);maxgap=max(maxgap,gap)
            independent.append(dict(name=f"independent_t{a['requested_t']}",elapsed_s=a['actual_relative_s'],
                exposure_unix_s=a['exposure_unix_s'],q_rad=q.tolist(),base_from_link6=kin.fk(q).tolist(),corners_uv=a['corners_uv']))
    plane=make_plane(args.transfer/'playground_grasp_before_01',cfg,kin)
    cube=cfg['calibration_objects'][0]
    initial=np.r_[np.zeros(6),cube['center_xy'],np.deg2rad(cube['yaw_deg']),.03]
    if args.fit_table_height:initial=np.r_[initial,0.]
    finger=make_finger_constraint(args.transfer,prior,K,args.finger_mode) if args.finger_constraint else None
    solved,observability,_=fit_model(K,prior,training,plane,initial,finger=finger)
    E,points=unpack(solved.x,prior)
    report=dict(status='Review-only offline candidate; no hardware commands',
        optimization_success=bool(solved.success),message=solved.message,
        reproduction=dict(argv=sys.argv,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest()),
        model_options=dict(finger_constraint=args.finger_constraint,finger_mode=args.finger_mode,fit_table_height=args.fit_table_height),
        validation_policy='Contiguous image block excluded from fitting, but model variants were compared on these metrics. These are reserved-image validation, not a completely blind final test.',
        fixed_intrinsics=K.tolist(),split={'train':[o['name'] for o in training],
            'heldout_contiguous':[o['name'] for o in holdout],'independent_annotator':[o['name'] for o in independent]},
        observed_samples=observations,independent_observed_samples=independent,
        time_alignment={'source':'D435i global_time device_timestamp_ms/1000; linearly interpolate measured encoder feedback',
            'max_interpolation_interval_s':maxgap,'optimized_time_offset':False},
        original={name:metric_errors(initial,prior,K,obs) for name,obs in [('train',training),('heldout_contiguous',holdout),('independent',independent)]},
        calibrated={name:metric_errors(solved.x,prior,K,obs) for name,obs in [('train',training),('heldout_contiguous',holdout),('independent',independent)]},
        plane_before=plane_error(initial,prior,plane),plane_after=plane_error(solved.x,prior,plane),
        plane_measurements=plane,parameters=solved.x.tolist(),
        table_height_relative_base_m=float(solved.x[10]) if len(solved.x)>10 else 0.,
        finger_constraint=None if finger is None else dict(reference=finger['reference'],
            before_residual_px=finger_residual(K,prior,finger).tolist(),
            after_residual_px=finger_residual(K,E,finger).tolist()),
        camera_delta_rotation_deg=np.degrees(solved.x[:3]).tolist(),camera_delta_translation_m=solved.x[3:6].tolist(),
        new_link6_from_camera_usd=(E@D).tolist(),cube_center_xy=solved.x[6:8].tolist(),
        cube_yaw_deg=float(np.degrees(solved.x[8])),cube_edge_m=float(solved.x[9]),
        observability=observability,
        limitations=['URDF joint zero and linkage geometry are held fixed, not independently calibrated.',
            'base_link/tabletop z=0 is a scene convention; physical mounting height remains an assumption.',
            'Only one motion path and one static cube; weak directions are regularized toward prior.',
            'Plane residual is a fitting constraint and is not independent validation.',
            'Post-contact cube position is excluded. Post-contact wrist depth is invalid at near range and excluded.',
            'No collision labels or grasp success scores enter the fit; no grasp/limit changes.',
            'Finger silhouette is approximate, piecewise smooth; local numerical Jacobian rank is not a physical covariance estimate.',
            'Default annotations, size prior and masks are specific to this stationary green cube sequence.'])
    candidate=copy.deepcopy(cfg);candidate['wrist_camera']['link6_from_camera_usd']=(E@D).tolist()
    candidate['wrist_camera']['status']='Offline multi-pose RGB + initial RGB-D fit; fixed live K; conditional on URDF zeros and base/table z=0; review-only candidate'
    green=candidate['calibration_objects'][0];green['center_xy']=solved.x[6:8].tolist();green['size']=[float(solved.x[9])]*3;green['yaw_deg']=float(np.degrees(solved.x[8]))
    # Re-fit global camera from fixed corrected cube, same initial global pixels and depth plane.
    gi=cfg['up_camera'];gplane=fit_plane(args.transfer/'playground_grasp_before_01/global_depth.npz',
                  json.loads((args.transfer/'playground_grasp_before_01/capture.json').read_text())['cameras']['global']['color'],'global')
    gn=np.array(gplane['normal']);gd=gplane['distance_m'];global_uv=np.array([[166,198],[195,199],[190,223],[160,222]],float)
    camera_points=np.array([ray(gi,uv)*(gd-solved.x[9])/(gn@ray(gi,uv)) for uv in global_uv])
    oldglobal=np.array(gi['base_from_camera_usd'])@D;R0=oldglobal[:3,:3]
    table_z=solved.x[10] if len(solved.x)>10 else 0.
    global_height=gd+table_z
    def global_res(v):
        R=Rotation.from_euler('z',v[0]).as_matrix()@R0
        return (((R@camera_points.T).T+[v[1],v[2],global_height])-points).ravel()
    gs=least_squares(global_res,[0,*oldglobal[:2,3]])
    G=np.eye(4);G[:3,:3]=Rotation.from_euler('z',gs.x[0]).as_matrix()@R0;G[:3,3]=[*gs.x[1:],global_height]
    candidate['up_camera'].update(base_from_camera_usd=(G@D).tolist(),eye_base_m=G[:3,3].tolist(),
        target_base_m=(G[:3,3]+.5*G[:3,2]).tolist(),status='Depth plane and corrected static green cube; dependent on multi-pose wrist registration')
    blue_uv=np.array([[459,177],[490,180],[488,201],[456,198]],float)
    bp=np.array([G[:3,:3]@(ray(gi,uv)*(gd-.03)/(gn@ray(gi,uv)))+G[:3,3] for uv in blue_uv])
    edge=((bp[3]-bp[0])+(bp[2]-bp[1]))/2
    candidate['calibration_objects'][1]['center_xy']=bp[:,:2].mean(0).tolist();candidate['calibration_objects'][1]['yaw_deg']=float(np.degrees(np.arctan2(edge[1],edge[0])))
    report['global_refit']=dict(eye_base_m=G[:3,3].tolist(),coordinate_rms_m=float(np.sqrt(np.mean(global_res(gs.x)**2))),
        dependency='Uses corrected cube and independently measured global depth plane, not an independent base/world calibration')
    if len(solved.x)>10:
        candidate['calibration_table_z_relative_base_m']=float(table_z)
    candidate.setdefault('reconstruction',{})['multiview_revision']=dict(report=str(args.output/'report.json'),
        status='Review candidate only; keep prior evidence',physical_accuracy='not measured absolute tool clearance')
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (args.output/'candidate_config.json').write_text(json.dumps(candidate,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['camera_delta_rotation_deg','camera_delta_translation_m','cube_center_xy','cube_edge_m','observability']},indent=2,allow_nan=False))
    print('Pixels:',json.dumps({name:{s:report[name][s]['point_rmse_px'] for s in ['train','heldout_contiguous','independent']} for name in ['original','calibrated']},allow_nan=False))


if __name__=='__main__':
    main()
