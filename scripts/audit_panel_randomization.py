#!/usr/bin/env python3
"""Independently audit all four XY panel corners from saved PhysX evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from audit_camera_sync import feedback_pixel_mask, project, quaternion_matrix
from audit_lerobot import IndependentFK
from audit_policy_eval import lamp_evidence, sha

ROOT = Path(__file__).resolve().parents[1]


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def inspect_corner(directory, cfg, offsets, calibrations):
    reported = json.loads((directory / 'report.json').read_text())
    x, y = offsets
    require(reported['panel_offset_x_m'] == x and reported['panel_offset_y_m'] == y, 'Corner XY identity mismatch')
    with np.load(directory / 'physics.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    n = len(arrays['q_actual'])
    require(n == reported['physics_steps'] and n > 2, 'Invalid physical trajectory length')
    for key, width in (('q_actual', 6), ('q_command', 6), ('gripper_actual', 2),
                       ('button_travel', 12), ('contact_force', 12), ('lights', 12)):
        require(arrays[key].shape == (n, width) and np.isfinite(arrays[key]).all(), f'Invalid {key} evidence')
    lamps, events = lamp_evidence(arrays['button_travel'], arrays['contact_force'], cfg['press_threshold'], cfg['release_threshold'])
    require(np.array_equal(lamps, arrays['lights']), 'Lights disagree with physical force/travel hysteresis')
    require([(event['type'], event['floor']) for event in events] == [(kind, floor) for floor in range(24, 36)
                                                                   for kind in ('pressed', 'released')], 'Missing/extra/wrong physical button events')
    require([(event['type'], event['floor'], event['physics_index']) for event in events] ==
            [(event['type'], event['floor'], event['step']) for event in reported['events']], 'Reported events disagree with raw physics')
    require(not lamps[[0, -1]].any() and np.max(np.abs(arrays['button_travel'][[0, -1]])) <= cfg['release_threshold'], 'Button reset/initial light mismatch')
    home_error = np.max(np.abs(arrays['q_actual'] - cfg['home_q']), axis=1)
    require(np.max(home_error[[0, -1]]) < cfg['home_tolerance_rad'], 'Initial/final arm is not folded home')
    returns = []
    for i in range(11):
        release, next_press = events[2 * i + 1]['physics_index'], events[2 * i + 2]['physics_index']
        minimum = float(home_error[release:next_press + 1].min())
        require(minimum < cfg['home_tolerance_rad'], f'No measured folded-home return after floor {24 + i}')
        returns.append(minimum)
    tracking = float(np.max(np.abs(arrays['q_actual'] - arrays['q_command'])))
    gripper = float(np.max(np.abs(arrays['gripper_actual'] - cfg['gripper_joint_positions_m'])))
    require(tracking < .15 and gripper < .00025, 'Excessive arm/gripper tracking error')
    require(abs(tracking - reported['max_joint_error_rad']) < 1e-9 and
            abs(home_error[-1] - reported['final_home_error_rad']) < 1e-9, 'Reported joint/home errors differ from physics')
    require(not reported['unexpected_collisions'], 'PhysX monitor recorded an unintended collision')
    # Cap YZ positions were not retained at each tick; keep this as reported
    # diagnostics rather than claiming independent lateral-motion verification.
    lateral = float(reported['max_button_lateral_error_m'])
    require(np.isfinite(lateral) and lateral < .0002, 'Reported button lateral drift exceeds threshold')
    fk = IndependentFK(ROOT / cfg['robot_urdf'])
    command = arrays['q_command']
    require(np.all(command >= fk.lower - 1e-7) and np.all(command <= fk.upper + 1e-7), 'Commanded joints exceed URDF position limits')
    xml = ET.parse(ROOT / cfg['robot_urdf']).getroot()
    velocity = np.array([float(xml.find(f"joint[@name='joint{i}']/limit").get('velocity')) for i in range(1, 7)])
    maximum_velocity = np.max(np.abs(np.diff(command, axis=0)), axis=0) / cfg['physics_dt']
    require(np.all(maximum_velocity <= velocity + 1e-6), 'Commanded joints exceed URDF speed limits')
    layout = reported['panel_layout']
    require(layout['panel_offset_x_m'] == x and layout['panel_offset_y_m'] == y and layout['fixed_rig_unchanged'] is True, 'Panel or fixed rig identity mismatch')
    require(set(layout['button_rest_positions_world_m']) == {str(floor) for floor in range(24, 36)}, 'Missing button rest measurements')
    rest_errors = []
    for floor in range(24, 36):
        expected = np.array([cfg['button_face_x'] + .003 + x,
                             cfg['button_column_y'] * (1 if floor < 30 else -1) + y,
                             cfg['button_bottom_z'] + ((floor - 24) % 6) * cfg['button_pitch_z']])
        rest_errors.append(float(np.linalg.norm(np.asarray(layout['button_rest_positions_world_m'][str(floor)]) - expected)))
    require(max(rest_errors) < .0002 and abs(max(rest_errors) - layout['max_button_rest_error_m']) < 1e-7,
            'Measured button rest locations disagree with XY corner')
    require(0 <= layout['max_spring_anchor_error_m'] <= .0002, 'Spring anchors did not move with panel')
    wall = np.asarray(layout['back_wall_world_bounds_m'], dtype=float)
    wall_error = abs(float(wall[0, 0]) - (cfg.get('wall_x', cfg['button_face_x'] + .06) + x))
    require(wall.shape == (2, 3) and np.isfinite(wall).all() and wall_error <= 1e-6,
            'Moved panel/back wall misalignment')
    require(abs(wall_error - layout['back_wall_alignment_error_m']) <= 1e-9, 'Incorrect wall alignment diagnostic')
    bounds = np.asarray(layout['panel_world_bounds_m'], dtype=float)
    require(bounds.shape == (2, 3) and np.isfinite(bounds).all() and np.all(bounds[0] < bounds[1]), 'Invalid whole-panel world bounds')
    require(abs(bounds[:, 1].mean() - y) < .0002, 'Panel frame did not move to corner Y')
    camera = layout['global_camera']
    require(camera['coverage_kind'] == 'frustum_only' and camera['resolution'] == [640, 480] and camera['all_inside'], 'Invalid camera coverage evidence')
    fixed = np.asarray(calibrations['global']['sensor']['world_optical_transform'])
    k_global = np.asarray(calibrations['global']['K'])
    require(np.allclose(fixed[:3, 3], cfg['global_camera_eye'], atol=1e-8), 'Global calibration does not match configured fixed camera')
    corners = np.asarray(list(product(*zip(bounds[0], bounds[1]))))
    pixels, depths = project(corners, fixed, k_global)
    projection_error = float(np.max(np.abs(pixels - camera['corners_px'])))
    require(projection_error < .001 and np.allclose(depths, camera['corner_depths_m'], atol=1e-6, rtol=0), 'Actual panel projection disagrees with independent fixed-camera projection')
    margin = cfg['panel_randomization']['camera_margin_px']
    require(camera['margin_px'] == margin and np.all(depths > 0) and np.all(pixels >= margin) and
            np.all(pixels <= np.array([639, 479]) - margin), 'Full panel is outside fixed-camera image margin')
    images, light_images = {}, []
    wrist_sensor = calibrations['wrist']['sensor']
    wrist_local = np.eye(4)
    wrist_local[:3, :3] = quaternion_matrix(wrist_sensor['optical_quaternion_wxyz_link6'])
    wrist_local[:3, 3] = wrist_sensor['optical_position_link6']
    for path in sorted(directory.glob('*.png')):
        with Image.open(path) as source:
            rgb = np.asarray(source.convert('RGB'))
        require(rgb.shape == (480, 640, 3) and rgb.std() > 2, 'Invalid selected RGB image')
        images[path.name] = sha(path)
    expected_images = {f'{prefix}_{view}.png' for prefix in ('home', 'pressed_24', 'pressed_29', 'pressed_30', 'pressed_35')
                       for view in ('global', 'wrist')}
    require(set(images) == expected_images, 'Missing selected camera evidence')
    for floor in (24, 29, 30, 35):
        step = next(event['physics_index'] for event in events if event['type'] == 'pressed' and event['floor'] == floor)
        link6 = fk.batch(arrays['q_actual'][step:step + 1])[0]
        link6[:3, 3] -= link6[:3, :3] @ [0., 0., .1358]
        link6[:3, 3] += [cfg['robot_base_x'], cfg['robot_base_y'], cfg['table_height']]
        center = [cfg['button_face_x'] + x + arrays['button_travel'][step, floor - 24],
                  cfg['button_column_y'] * (1 if floor < 30 else -1) + y,
                  cfg['button_bottom_z'] + ((floor - 24) % 6) * cfg['button_pitch_z']]
        target = np.array(center) + np.array([[0., dy, dz] for dy, dz in product((-.013, .013), repeat=2)])
        counts = {}
        for view in ('global', 'wrist'):
            image = np.asarray(Image.open(directory / f'pressed_{floor}_{view}.png').convert('RGB'))
            uv, depth = project(target, fixed if view == 'global' else link6 @ wrist_local, np.asarray(calibrations[view]['K']))
            lo = np.maximum(np.floor(uv.min(0)).astype(int) - 2, [0, 0])
            hi = np.minimum(np.ceil(uv.max(0)).astype(int) + 3, [640, 480])
            require(np.all(depth > 0) and np.all(hi > lo), 'Lit button projection is invalid')
            counts[view] = int(feedback_pixel_mask(image[lo[1]:hi[1], lo[0]:hi[0]]).sum())
        require(counts['wrist'] >= 20, f'Wrist image does not show pressed floor {floor} orange feedback')
        light_images.append(dict(floor=floor, physics_index=step, amber_pixels=counts))
    require(reported['success'] is True, 'Corner report rejected its own physical result')
    return dict(corner=reported['corner'], success=True, panel_offset_x_m=x, panel_offset_y_m=y, physics_steps=n,
                physical_presses=12, physical_releases=12, events=events,
                initial_home_error_rad=float(home_error[0]), final_home_error_rad=float(home_error[-1]),
                between_button_home_error_rad=returns, max_joint_tracking_error_rad=tracking,
                max_gripper_error_m=gripper, max_command_joint_velocity_rad_s=maximum_velocity.tolist(),
                max_measured_button_rest_error_m=max(rest_errors), max_independent_projection_error_px=projection_error,
                minimum_full_panel_margin_px=float(np.minimum(pixels, np.array([639, 479]) - pixels).min()),
                reported_max_button_lateral_error_m=lateral, unexpected_collisions=0,
                selected_light_images=light_images, selected_images_sha256=images,
                physics_sha256=sha(directory / 'physics.npz'), report_sha256=sha(directory / 'report.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--snapshot-directory', type=Path, default=ROOT / 'outputs/edge_feedback')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    require(not args.report.exists(), 'Refusing to overwrite completed audit evidence')
    recorded = json.loads((args.run / 'report.json').read_text())
    cfg = recorded['config']
    require(recorded['success'] is True and len(recorded['corners']) == 4, 'Four physical corner runs have not completed')
    r = cfg['panel_randomization']
    offsets = list(product((r['min_offset_x_m'], r['max_offset_x_m']), (r['min_offset_y_m'], r['max_offset_y_m'])))
    require(len(set(offsets)) == 4, 'Four distinct XY extremes are required')
    calibrations = {view: json.loads((args.snapshot_directory / f'{view}_camera/intrinsics.json').read_text()) for view in ('global', 'wrist')}
    report = dict(success=False, audited_at=datetime.now(timezone.utc).isoformat(), run=str(args.run.resolve()),
                  config=cfg, corners=[], errors=[], audit_source_sha256=sha(Path(__file__)),
                  calibration_sha256={view: sha(args.snapshot_directory / f'{view}_camera/intrinsics.json') for view in calibrations},
                  limitations=['Frustum coverage checks complete panel bounds; it does not prove absence of occlusion.',
                               'Unexpected collision classification and spring anchors use the recorded PhysX/USD monitor evidence.',
                               'Per-tick cap YZ coordinates were not retained; lateral drift is reported diagnostics only.',
                               'Only selected still RGB frames were recorded; this is not a dataset or continuous RGB trajectory.'])
    for index, xy in enumerate(offsets):
        try:
            corner = inspect_corner(args.run / f'corner_{index}', cfg, xy, calibrations)
            report['corners'].append(corner)
        except Exception as error:
            report['errors'].append(dict(corner=index, error=f'{type(error).__name__}: {error}'))
    report.update(success=not report['errors'] and len(report['corners']) == 4,
                  physical_presses=sum(corner['physical_presses'] for corner in report['corners']))
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in ('success', 'physical_presses', 'errors')}, indent=2))
    raise SystemExit(0 if report['success'] else 1)


if __name__ == '__main__':
    main()
