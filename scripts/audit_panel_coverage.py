#!/usr/bin/env python3
"""Audit stratified XY coverage from committed raw metadata and measured layouts.

This does not import the planner, sampler, collector or shared panel validator.
Physics success and RGB semantics are audited by the separate raw/prefix audits.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import re

import numpy as np


FLOORS = tuple(range(24, 36))
SOURCE_FILES = ('frames.npz', 'physics.npz', 'wrist.mp4', 'global.mp4')


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def number(value, name):
    require(type(value) in (float, int) and np.isfinite(value), f'Invalid finite number: {name}')
    return float(value)


def close(actual, expected, name, tolerance=1e-12):
    values = np.asarray(actual, dtype=float)
    wanted = np.asarray(expected, dtype=float)
    require(values.shape == wanted.shape and np.isfinite(values).all()
            and np.allclose(values, wanted, rtol=0, atol=tolerance), f'Inconsistent {name}')


def grid_settings(collection):
    require(collection['raw_schema_version'] == 11 and collection['fps'] == 30
            and collection['physics_hz'] == 120 and collection['capture_stride'] == 4,
            'Coverage audit requires schema 11, 30 Hz capture and 120 Hz physics')
    settings = collection['config']['panel_randomization']
    require(settings['enabled'] is True and settings['sampling_mode'] == 'stratified_grid'
            and settings['grid_shape'] == [10, 10] and settings['boundary_mode'] == 'corners',
            'Expected enabled 10 x 10 per-floor stratification including exact corners')
    require(type(settings['grid_seed']) is int and settings['grid_seed'] >= 0, 'Invalid grid seed')
    for axis in 'xy':
        lo, hi = (number(settings[f'{bound}_offset_{axis}_m'], f'{bound} {axis}') for bound in ('min', 'max'))
        require(lo < hi, f'Empty {axis} range')
    return settings


def offset_cell(value, lo, hi, bins):
    """Half-open cells, with the last cell including the exact outer maximum."""
    require(lo <= value <= hi, 'Offset is outside the configured rectangle')
    return min(int(np.floor((value - lo) / (hi - lo) * bins)), bins - 1)


def project_bounds(bounds, collection, env_offset):
    calibration = collection['global']
    require((calibration['width'], calibration['height']) == (640, 480), 'Wrong global camera resolution')
    camera = np.asarray(calibration['sensor']['world_optical_transform'], dtype=float)
    intrinsics = np.asarray(calibration['K'], dtype=float)
    require(camera.shape == (4, 4) and intrinsics.shape == (3, 3)
            and np.isfinite(camera).all() and np.isfinite(intrinsics).all(), 'Invalid fixed camera calibration')
    cfg = collection['config']
    close(camera[:3, 3], cfg['global_camera_eye'], 'fixed camera eye', 1e-8)
    forward = np.asarray(cfg['global_camera_target']) - camera[:3, 3]
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    close(camera[:3, :3], np.column_stack((right, np.cross(right, forward), -forward)), 'fixed camera rotation', 1e-8)
    corners = np.array(list(product(*zip(bounds[0], bounds[1]))))
    local = (corners - np.asarray(env_offset) - camera[:3, 3]) @ camera[:3, :3]
    depth = -local[:, 2]
    require(np.all(depth > .005), 'Panel bounds are behind or too near the fixed camera')
    pixels = np.column_stack((intrinsics[0, 0] * local[:, 0] / depth + intrinsics[0, 2],
                              intrinsics[1, 2] - intrinsics[1, 1] * local[:, 1] / depth))
    return pixels, depth


def inspect_metadata(metadata, collection):
    settings = grid_settings(collection)
    cfg = collection['config']
    eid, floor = metadata['episode_id'], metadata['floor']
    require(type(eid) is int and 0 <= eid < 1200 and type(floor) is int and floor == 24 + eid % 12,
            'Episode ID does not identify its floor within the 1200-episode schedule')
    require(metadata['success'] is True and metadata['task'] == f'Press {floor} floor.', 'Unsuccessful or wrong task record')
    require(metadata['collection_fingerprint'] == collection['collection_fingerprint'], 'Episode collection fingerprint differs')
    for key in ('raw_schema_version', 'fps', 'physics_hz', 'capture_stride', 'action_horizon_s'):
        require(metadata[key] == collection[key], f'Episode {key} differs from collection')
    require(metadata['seed'] == collection['seed'] + eid, 'Episode seed does not match collection schedule')
    xy = np.array([number(metadata[f'panel_offset_{axis}_m'], axis) for axis in 'xy'])
    variation, layout = metadata['variation'], metadata['panel_layout']
    proof = variation['panel_randomization']
    for value, label in ((variation, 'variation'), (layout, 'measured layout'), (proof, 'sampling proof'),
                         (proof['validation'], 'layout validation')):
        close([value[f'panel_offset_{axis}_m'] for axis in 'xy'], xy, label)
    if 'variation' in variation:
        close([variation['variation'][f'panel_offset_{axis}_m'] for axis in 'xy'], xy, 'motion variation')
    require(proof['enabled'] is True and proof['axis'] == 'world_xy'
            and proof['distribution'] == 'stratified_grid_with_safety_rejection', 'Wrong panel sampling distribution')
    for key in ('sampling_mode', 'grid_shape', 'grid_seed', 'boundary_mode'):
        require(proof[key] == settings[key], f'Sampling {key} differs from configuration')
    stream = [settings['grid_seed'], floor, 0x47524944]
    require(proof['grid_permutation_seed_stream'] == stream
            and proof['grid_permutation_order'] == 'flat_x_major_y_minor', 'Wrong grid permutation convention')
    ordinal = eid // 12
    require(type(proof['episode_index']) is int and proof['episode_index'] == ordinal, 'Wrong within-floor episode index')
    expected_flat = int(np.random.default_rng(np.random.SeedSequence(stream)).permutation(100)[ordinal])
    cell = [expected_flat // 10, expected_flat % 10]
    require(proof['grid_flat_index'] == expected_flat and proof['grid_cell_index_xy'] == cell,
            'Cell label differs from the independently reconstructed permutation')
    bounds = {}
    derived_cell = []
    for i, axis in enumerate('xy'):
        lo, hi = settings[f'min_offset_{axis}_m'], settings[f'max_offset_{axis}_m']
        close([proof[f'min_offset_{axis}_m'], proof[f'max_offset_{axis}_m']], [lo, hi], f'configured {axis} range')
        edges = np.linspace(lo, hi, 11)
        bounds[axis] = edges[cell[i]:cell[i] + 2]
        close(proof['grid_cell_bounds_m'][axis], bounds[axis], f'{axis} cell bounds')
        derived_cell.append(offset_cell(xy[i], lo, hi, 10))
    require(derived_cell == cell, 'Actual offset belongs to another grid cell')
    corner = cell[0] in (0, 9) and cell[1] in (0, 9)
    require(type(proof['boundary_corner']) is bool and proof['boundary_corner'] == corner, 'Incorrect corner designation')
    if corner:
        for i, axis in enumerate('xy'):
            wanted = settings[f'{"min" if cell[i] == 0 else "max"}_offset_{axis}_m']
            require(xy[i] == wanted, 'Required corner is not at the exact outer rectangle boundary')
    require(proof['seed_stream'] == [metadata['seed'], floor, 0x50414E45], 'Wrong layout jitter seed stream')
    rejected, attempts = proof['rejected_candidates'], proof['attempts']
    require(type(attempts) is int and 1 <= attempts <= settings['max_attempts']
            and len(rejected) == attempts - 1, 'Invalid rejection-sampling attempt count')
    require(not corner or attempts == 1, 'Required corner was resampled')
    for candidate in rejected:
        require(isinstance(candidate['reason'], str) and bool(candidate['reason']), 'Missing rejection reason')
        for i, axis in enumerate('xy'):
            value = number(candidate[f'panel_offset_{axis}_m'], 'rejected candidate')
            require(bounds[axis][0] <= value <= bounds[axis][1], 'Rejected candidate escaped its assigned cell')

    env = np.asarray(metadata['env_offset_m'], dtype=float)
    require(env.shape == (3,) and np.isfinite(env).all(), 'Invalid environment offset')
    close(metadata['robot_base_world_m'], env + [cfg['robot_base_x'], cfg['robot_base_y'], cfg['table_height']], 'fixed robot base')
    require(layout['fixed_rig_unchanged'] is True, 'Panel movement changed the fixed rig')
    rests = layout['button_rest_positions_world_m']
    require(set(rests) == {str(f) for f in FLOORS}, 'Missing measured button rests')
    rest_errors, recovered = [], []
    for f in FLOORS:
        nominal = np.array([cfg['button_face_x'] + .003, cfg['button_column_y'] * (1 if f < 30 else -1),
                            cfg['button_bottom_z'] + ((f - 24) % 6) * cfg['button_pitch_z']])
        measured = np.asarray(rests[str(f)], dtype=float)
        close(measured, nominal + env + [xy[0], xy[1], 0.], f'actual button {f} rest', .0002)
        rest_errors.append(float(np.linalg.norm(measured - nominal - env - [xy[0], xy[1], 0.])))
        recovered.append((measured - nominal - env)[:2])
    require(max(rest_errors) <= .0002, 'Measured panel rest error exceeds tolerance')
    close(layout['max_button_rest_error_m'], max(rest_errors), 'reported rest error', 1e-7)
    require(0 <= number(layout['max_spring_anchor_error_m'], 'spring anchor error') <= .0002, 'Spring anchor movement error')
    panel = np.asarray(layout['panel_world_bounds_m'], dtype=float)
    require(panel.shape == (2, 3) and np.all(panel[0] < panel[1]), 'Invalid panel bounds')
    bottom, top = cfg['button_bottom_z'] - .07, cfg['button_bottom_z'] + 5 * cfg['button_pitch_z'] + .07
    close(panel - env, [[cfg['button_face_x'] + xy[0] - .000225, xy[1] - .1, bottom],
                        [cfg['button_face_x'] + xy[0] + .049, xy[1] + .1, top]], 'shifted whole-panel bounds', .0002)
    wall = np.asarray(layout['back_wall_world_bounds_m'], dtype=float)
    require(wall.shape == (2, 3) and np.isfinite(wall).all() and np.all(wall[0] < wall[1]), 'Invalid back wall bounds')
    wall_error = abs(wall[0, 0] - (cfg.get('wall_x', cfg['button_face_x'] + .06) + xy[0] + env[0]))
    require(wall_error <= 1e-6, 'Back wall does not follow panel X')
    close(layout['back_wall_alignment_error_m'], wall_error, 'back wall alignment diagnostic', 1e-9)
    pixels, depths = project_bounds(panel, collection, env)
    camera = layout['global_camera']
    require(camera['resolution'] == [640, 480] and camera['all_inside'] is True
            and camera['coverage_kind'] == 'frustum_only', 'Invalid measured camera coverage')
    close(camera['corners_px'], pixels, 'independent panel projection', .001)
    close(camera['corner_depths_m'], depths, 'independent camera depths', 1e-6)
    margin = float(np.minimum(pixels, np.array([639, 479]) - pixels).min())
    require(camera['margin_px'] == settings['camera_margin_px'] and margin >= settings['camera_margin_px'], 'Full panel lacks required image margin')
    validation = proof['validation']
    require(validation['all_buttons_reachable'] is True and validation['reachable_floors'] == list(FLOORS),
            'Layout did not record successful planning checks for all twelve floors')
    planned_camera = validation['camera_coverage']
    require(planned_camera['all_inside'] is True and planned_camera['required_margin_px'] == settings['camera_margin_px']
            and planned_camera['minimum_margin_px'] >= settings['camera_margin_px'], 'Planning camera check failed')
    clearance_records = {}
    for key, check in (('all_floor_paths', validation['trajectory_validation']), ('actual_episode_path', variation['validation'])):
        require(type(check['physics_samples_checked']) is int and check['physics_samples_checked'] > 1, 'Missing trajectory samples checked')
        require(0 <= number(check['maximum_fk_target_error_m'], 'planned FK error') <= .0002, 'Excessive planned FK error')
        clearances = check['conservative_clearances']
        for obstacle, minimum in (('table_m', .005), ('wall_or_panel_m', .005), ('fixed_camera_m', .015)):
            require(number(clearances[obstacle], obstacle) >= minimum, f'Unsafe recorded {key} {obstacle}')
        clearance_records[key] = dict(clearances)
    return dict(episode_id=eid, floor=floor, episode_index=ordinal, grid_flat_index=expected_flat,
                grid_cell_index_xy=cell, boundary_corner=corner, panel_offset_x_m=float(xy[0]), panel_offset_y_m=float(xy[1]),
                measured_panel_offset_xy_m=np.mean(recovered, axis=0).tolist(), maximum_button_rest_error_m=max(rest_errors),
                minimum_actual_camera_margin_px=margin, maximum_projection_error_px=float(np.abs(pixels - camera['corners_px']).max()),
                all_floors_checked=list(FLOORS), recorded_conservative_clearances=clearance_records, attempts=attempts)


def coverage_counts(rows, allow_partial=False):
    require(len({r['episode_id'] for r in rows}) == len(rows), 'Duplicate episode IDs')
    result = {}
    for floor in FLOORS:
        selected = [r for r in rows if r['floor'] == floor]
        count = Counter(tuple(r['grid_cell_index_xy']) for r in selected)
        require(all(value == 1 for value in count.values()), f'Floor {floor} has repeated grid cells')
        require(len({r['episode_index'] for r in selected}) == len(selected), f'Floor {floor} has duplicate within-floor indices')
        corners = [r for r in selected if r['boundary_corner']]
        if not allow_partial:
            require(len(selected) == 100 and len(count) == 100 and len(corners) == 4
                    and {r['episode_index'] for r in selected} == set(range(100)), f'Floor {floor} lacks full 100-cell/four-corner coverage')
        occupancy = np.zeros((10, 10), dtype=int)
        for cell, value in count.items():
            occupancy[cell] = value
        result[str(floor)] = dict(episodes=len(selected), occupied_cells=len(count), exact_corners=len(corners),
                                  occupancy_x_y=occupancy.tolist())
    return result


def audit(raw, allow_partial=False):
    raw = Path(raw).resolve()
    collection_path = raw / 'collection_metadata.json'
    collection = json.loads(collection_path.read_text())
    settings = grid_settings(collection)
    identity = collection['identity']
    require(all(collection[key] == value for key, value in identity.items()), 'Collection fields differ from its identity')
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    require(collection['collection_fingerprint'] == fingerprint, 'Collection fingerprint does not match serialized identity')
    require(sha(collection['source_snapshot']) == collection['scene_sha256'], 'Frozen snapshot changed')
    for view in ('wrist', 'global'):
        record = collection['source_calibrations'][view]
        require(sha(record['path']) == record['sha256'] == collection['camera_calibration_sha256'][view], 'Frozen camera calibration changed')
    report = dict(success=False, raw=str(raw), audited_at=datetime.now(timezone.utc).isoformat(), allow_partial=allow_partial,
        collection_fingerprint=fingerprint, collection_metadata_sha256=sha(collection_path), scene_sha256=collection['scene_sha256'],
        audit_source_sha256=sha(Path(__file__)), settings=settings, episodes=[], errors=[],
        limitations=['Coverage and measured static layout are checked independently; run raw physics and press-prefix audits for task outcomes and action/RGB alignment.',
                     'Camera frustum coverage does not establish absence of temporary robot occlusion.',
                     'All-floor trajectory clearances and spring-anchor errors are recorded planner/USD evidence, not trajectories regenerated by this audit.'])
    for directory in sorted(raw.glob('episode_*')):
        if not directory.is_dir() or not re.fullmatch(r'episode_[0-9]{6}', directory.name):
            continue
        try:
            path = directory / 'metadata.json'
            metadata = json.loads(path.read_text())
            require(directory.name == f"episode_{metadata['episode_id']:06d}", 'Directory episode ID differs')
            row = inspect_metadata(metadata, collection)
            require(set(metadata['source_files']) == set(SOURCE_FILES), 'Incomplete raw source-file identity')
            identities = {'metadata.json': dict(sha256=sha(path), bytes=path.stat().st_size)}
            for name in SOURCE_FILES:
                path = directory / name
                record = dict(sha256=sha(path), bytes=path.stat().st_size)
                require(record == metadata['source_files'][name], f'Raw source changed: {name}')
                identities[name] = record
            row['source_files'] = identities
            report['episodes'].append(row)
        except Exception as exc:
            report['errors'].append(dict(episode=directory.name, error=f'{type(exc).__name__}: {exc}'))
    try:
        require(bool(report['episodes']), 'No committed stratified episodes')
        report['floor_coverage'] = coverage_counts(report['episodes'], allow_partial)
    except Exception as exc:
        report['errors'].append(dict(error=f'{type(exc).__name__}: {exc}'))
    report.update(total_episodes=len(report['episodes']), success=not report['errors'])
    if report['episodes']:
        report.update(minimum_actual_camera_margin_px=min(r['minimum_actual_camera_margin_px'] for r in report['episodes']),
            minimum_recorded_all_floor_camera_clearance_m=min(r['recorded_conservative_clearances']['all_floor_paths']['fixed_camera_m'] for r in report['episodes']),
            minimum_recorded_episode_camera_clearance_m=min(r['recorded_conservative_clearances']['actual_episode_path']['fixed_camera_m'] for r in report['episodes']))
    return report


def plot_coverage(report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    settings = report['settings']
    xmin, xmax = settings['min_offset_x_m'] * 1000, settings['max_offset_x_m'] * 1000
    ymin, ymax = settings['min_offset_y_m'] * 1000, settings['max_offset_y_m'] * 1000
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharex=True, sharey=True)
    for floor, axis in zip(FLOORS, axes.flat):
        rows = [r for r in report['episodes'] if r['floor'] == floor]
        for value in np.linspace(ymin, ymax, 11):
            axis.axvline(value, color='.85', lw=.6, zorder=0)
        for value in np.linspace(xmin, xmax, 11):
            axis.axhline(value, color='.85', lw=.6, zorder=0)
        if rows:
            axis.scatter([r['panel_offset_y_m'] * 1000 for r in rows], [r['panel_offset_x_m'] * 1000 for r in rows],
                         s=15, color='#1778a8', label='Recorded offset')
            corners = [r for r in rows if r['boundary_corner']]
            axis.scatter([r['panel_offset_y_m'] * 1000 for r in corners], [r['panel_offset_x_m'] * 1000 for r in corners],
                         s=38, color='#d95f02', marker='s', clip_on=False, label='Exact corner')
        occupied = len({tuple(r['grid_cell_index_xy']) for r in rows})
        axis.set(title=f'Floor {floor}: {len(rows)} episodes / {occupied} cells', xlim=(ymin, ymax), ylim=(xmin, xmax))
        axis.set_xlabel('Y offset (mm)')
        axis.set_ylabel('X offset (mm)')
    fig.suptitle(f"Recorded panel coverage: {report['total_episodes']} / 1200 episodes; audit {'PASS' if report['success'] else 'FAIL'}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw', type=Path)
    parser.add_argument('--report', '--output', dest='report', required=True, type=Path)
    parser.add_argument('--plot', type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    plot = args.plot or args.report.with_suffix('.png')
    require(not args.report.exists() and not plot.exists(), 'Choose fresh report and plot paths')
    report = audit(args.raw, args.allow_partial)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    plot.parent.mkdir(parents=True, exist_ok=True)
    plot_coverage(report, plot)
    report['plot'] = dict(path=str(plot.resolve()), sha256=sha(plot))
    with args.report.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: report[key] for key in ('success', 'total_episodes', 'errors')}, indent=2))
    raise SystemExit(0 if report['success'] else 1)


if __name__ == '__main__':
    main()
