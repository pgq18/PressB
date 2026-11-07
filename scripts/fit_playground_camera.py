#!/usr/bin/env python3
"""Fit the playground global camera from stationary RGB-D and a PAP wrist prior.

The DEFAULT hand-labelled cube corners are for playground_capture_01 recorded
2026-11-07 ONLY. Other captures/camera placements require new annotations via
--correspondences. This is approximate registration, not a general calibration
algorithm. A tabletop plane fixes tilt and height; a shared green cube fixes
in-plane rotation and translation relative to the PAP wrist extrinsic prior and measured wrist intrinsics.
The blue cube is a calibration-only prop and is never added to the clean scene.

Example (numpy/scipy required)::

    python scripts/fit_playground_camera.py \
      --capture outputs/playground_twin_20261107/playground_capture_01 \
      --output outputs/playground_twin_20261107/camera_refit.json

--correspondences JSON keys: green_global_uv (4x2), green_wrist_uv (4x2),
blue_global_uv (4x2), plate_edge_uv (2x2), cube_edge_m (positive number).
Green corners must correspond between views. Their order is global image
far-left, far-right, near-right, near-left. The plate edge supplies an initial
yaw direction only; neither its position nor its dimensions are assumed known.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pressb.robots.piper.kinematics import PiperKinematics

DEFAULT_ANNOTATIONS = {
    'green_global_uv': [[166, 198], [195, 199], [190, 223], [160, 222]],
    'green_wrist_uv': [[447, 220], [401, 226], [388, 187], [433, 181]],
    'blue_global_uv': [[459, 177], [490, 180], [488, 201], [456, 198]],
    'plate_edge_uv': [[213, 30], [405, 29]],
    'cube_edge_m': .03,
}


def ray(intrinsics, uv):
    u, v = uv
    return np.array([(u-intrinsics['cx'])/intrinsics['fx'],
                     (v-intrinsics['cy'])/intrinsics['fy'], 1.])


def fit_plane(depth_path, intrinsics, camera):
    depth = np.load(depth_path)['depth_m']
    if depth.shape != (intrinsics['height'], intrinsics['width']):
        raise ValueError('Aligned depth shape must match color intrinsics')
    vv, uu = np.indices(depth.shape)
    xyz = np.stack([(uu-intrinsics['cx'])/intrinsics['fx']*depth,
                    (vv-intrinsics['cy'])/intrinsics['fy']*depth, depth], -1)
    valid = np.isfinite(depth) & (depth > .1) & (depth < 2.)
    # Static capture-specific masks omit the base plate and near gripper.
    valid[:60 if camera == 'global' else 35] = False
    if camera == 'wrist':
        valid[350:, 480:] = False
    points = xyz[valid][::5]
    for _ in range(5):
        if len(points) < 100:
            raise ValueError('Too few table depth points after plane rejection')
        center = points.mean(0)
        _, _, vectors = np.linalg.svd(points-center, full_matrices=False)
        normal = vectors[-1]
        distance = normal @ center
        if distance < 0:
            normal, distance = -normal, -distance
        points = points[np.abs(points@normal-distance) < .004]
    return dict(normal=normal.tolist(), distance_m=float(distance),
                rms_m=float(np.sqrt(np.mean((points@normal-distance)**2))),
                points=len(points))


def project_to_plane(intrinsics, uv, normal, distance, height):
    direction = ray(intrinsics, uv)
    denominator = float(normal @ direction)
    if denominator <= 0 or distance <= height:
        raise ValueError('Annotated ray does not intersect table in front of camera')
    return direction * ((distance-height)/denominator)


def fit(capture_dir, pap_config, annotations, wrist_intrinsics="measured"):
    cap = json.loads((capture_dir/'capture.json').read_text())
    cfg = json.loads(pap_config.read_text())
    for key, count in [('green_global_uv',4),('green_wrist_uv',4),
                       ('blue_global_uv',4),('plate_edge_uv',2)]:
        value = np.asarray(annotations[key], float)
        if value.shape != (count,2) or not np.all(np.isfinite(value)):
            raise ValueError(f'{key} must contain {count} finite pixel pairs')
    edge = float(annotations['cube_edge_m'])
    if not np.isfinite(edge) or edge <= 0:
        raise ValueError('cube_edge_m must be positive')
    joints = np.array(cap['feedback']['median_joints_deg'], float)
    ranges = np.array(cap['feedback']['joint_range_deg'], float)
    if np.max(ranges) > .1:
        raise ValueError('This fit requires a stationary robot during both captures')
    intr = cap['cameras']['global']['color']
    planes = {name: fit_plane(capture_dir/f'{name}_depth.npz',
                             cap['cameras'][name]['color'],name)
              for name in ('global','wrist')}
    normal = np.array(planes['global']['normal'])
    distance = planes['global']['distance_m']
    urdf = Path(cfg['robot_urdf'])
    if not urdf.is_absolute():
        urdf = ROOT / urdf
    wrist = (PiperKinematics(urdf,tip_offset=0).fk(np.deg2rad(joints))
             @ np.asarray(cfg['wrist_camera']['link6_from_camera_usd'])
             @ np.diag([1.,-1.,-1.,1.]))
    a,b = [project_to_plane(intr,uv,normal,distance,0)
           for uv in annotations['plate_edge_uv']]
    base_y = (b-a)/np.linalg.norm(b-a)
    base_z = -normal
    base_x = np.cross(base_y,base_z)
    base_x /= np.linalg.norm(base_x)
    initial_rotation = np.stack([base_x,np.cross(base_z,base_x),base_z])
    global_points = np.array([project_to_plane(intr,uv,normal,distance,edge)
                              for uv in annotations['green_global_uv']])
    candidates = {}
    for label, wi in [('preserved_pap_wrist',cfg['wrist_camera']),
                      ('measured_wrist_intrinsics',cap['cameras']['wrist']['color'])]:
        wrist_points = []
        for uv in annotations['green_wrist_uv']:
            direction = wrist[:3,:3] @ ray(wi,uv)
            scale = (edge-wrist[2,3])/direction[2]
            if scale <= 0:
                raise ValueError('Wrist cube ray points away from assumed cube top')
            wrist_points.append(wrist[:3,3]+scale*direction)
        wrist_points = np.array(wrist_points)
        def residual(v):
            rotation = Rotation.from_euler('z',v[0]).as_matrix() @ initial_rotation
            predicted = (rotation@global_points.T).T + [v[1],v[2],distance]
            return (predicted-wrist_points).ravel()
        solved = least_squares(residual,[0.,.71,.05])
        if not solved.success:
            raise RuntimeError(solved.message)
        transform = np.eye(4)
        transform[:3,:3] = Rotation.from_euler('z',solved.x[0]).as_matrix() @ initial_rotation
        transform[:3,3] = [*solved.x[1:],distance]
        candidates[label] = dict(base_from_global_cv=transform.tolist(),
            base_from_global_usd=(transform@np.diag([1.,-1.,-1.,1.])).tolist(),
            rms_coordinate_m=float(np.sqrt(np.mean(residual(solved.x)**2))),
            yaw_correction_deg=float(np.degrees(solved.x[0])))
    selected_label = ('measured_wrist_intrinsics' if wrist_intrinsics == 'measured' else 'preserved_pap_wrist')
    selected = candidates[selected_label]
    transform = np.array(selected['base_from_global_cv'])
    calibration_objects = []
    for name,key in [('GreenCube','green_global_uv'),('BlueCube','blue_global_uv')]:
        points = np.array([transform[:3,:3]@project_to_plane(intr,uv,normal,distance,edge)
                           + transform[:3,3] for uv in annotations[key]])
        along_x = ((points[3]-points[0])+(points[2]-points[1]))/2
        calibration_objects.append(dict(name=name,shape='cube',edge=edge,
            size_xyz=[edge]*3,center_xy=points[:,:2].mean(0).tolist(),
            center_z_base_m=edge/2,
            yaw_deg=float(np.degrees(np.arctan2(along_x[1],along_x[0]))),
            status='Calibration-only prop; omitted from clean playground'))
    wn = np.array(planes['wrist']['normal'])
    wp = wrist[:3,:3]@(wn*planes['wrist']['distance_m'])+wrist[:3,3]
    return dict(
        status='Approximate static RGB-D registration with PAP wrist extrinsic prior',
        selected_wrist_intrinsics=wrist_intrinsics,selected_candidate=selected_label,
        capture=str(capture_dir),pap_wrist_prior=str(pap_config),
        annotation_scope='Default annotations and depth masks are specific to 2026-11-07 capture_01; reannotate other captures.',
        annotations=annotations,joint_degrees=joints.tolist(),depth_planes=planes,
        recommendation=dict(resolution=[intr['width'],intr['height']],
            **{k:intr[k] for k in ('fx','fy','cx','cy')},
            eye_base_m=transform[:3,3].tolist(),
            target_base_m=(transform[:3,3]+.5*transform[:3,2]).tolist(),
            base_from_camera_usd=selected['base_from_global_usd'],
            horizontal_fov_deg=float(np.degrees(2*np.arctan(intr['width']/2/intr['fx'])))),
        candidates=candidates,calibration_objects=calibration_objects,
        wrist_prior_diagnostic=dict(measured_intrinsics=cap['cameras']['wrist']['color'],
            preserved_PAP_intrinsics={k:cfg['wrist_camera'][k] for k in ('fx','fy','cx','cy')},
            fitted_table_closest_point_base_m=wp.tolist(),
            fitted_table_up_normal_base=(wrist[:3,:3]@(-wn)).tolist()),
        uncertainty=dict(xy='Allow approximately2 cm for approximate wrist/cube registration; no statistical interval established.',
            height='Plane residual excludes depth bias and robot-base/table offset; allow5-10 mm.',
            yaw='Manual corners/plate edge and approximate wrist transform; allow1-2 degrees.',
            wrist='Default uses measured hardware intrinsics and unchanged PAP link6 extrinsics. PAP focal length is approximately8% different. Depth-derived table offset of11.7 mm remains uncalibrated.',
            table_size='Table edges outside view: dimensions remain priors.',
            plate='Plate position and dimensions are not calibration measurements; only edge direction initializes rotation.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='Output JSON file')
    parser.add_argument('--pap-config',type=Path,default=ROOT/'configs/piper_pick_place.json')
    parser.add_argument('--correspondences',type=Path)
    parser.add_argument('--wrist-intrinsics',choices=('measured','pap'),default='measured',
                        help='Measured hardware intrinsics by default; pap retains old image prior for comparison')
    args = parser.parse_args()
    annotations = (json.loads(args.correspondences.read_text()) if args.correspondences
                   else DEFAULT_ANNOTATIONS)
    result = fit(args.capture,args.pap_config,annotations,args.wrist_intrinsics)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'eye_base_m':result['recommendation']['eye_base_m'],
                      'rms_coordinate_m':result['candidates'][result['selected_candidate']]['rms_coordinate_m']},indent=2))


if __name__ == '__main__':
    main()
