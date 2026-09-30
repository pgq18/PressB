"""Independent evidence tests for repeated-floor, shifted-panel evaluation reports."""
from copy import deepcopy
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_policy_eval import (episode_schedule, outcome_statistics, panel_episode_geometry,
                               panel_layout_plan, request_seed_evidence)


POSITIONS = [('center', 0., 0.), ('xmin_ymin', -.01, -.025), ('xmin_ymax', -.01, .025),
             ('xmax_ymin', .01, -.025), ('xmax_ymax', .01, .025)]


def manifest_fixture(floors=range(24, 36), repeats=1):
    cfg = dict(panel_randomization=dict(enabled=True, min_offset_x_m=-.01, max_offset_x_m=.01,
                                        min_offset_y_m=-.025, max_offset_y_m=.025))
    layouts = [dict(panel_layout_index=i, panel_layout_name=name, panel_offset_x_m=x, panel_offset_y_m=y)
               for i, (name, x, y) in enumerate(POSITIONS)]
    schedule = [dict(episode_id=ordinal, repeat=repeat, floor=floor, seed_episode_index=(repeat * 5 + li) * 12 + floor - 24,
                     **layouts[li])
                for ordinal, (repeat, li, floor) in enumerate(product(range(repeats), range(5), floors))]
    return dict(config=cfg, arguments=dict(panel_layouts='center_corners', floors=list(floors), episodes_per_floor=repeats),
                panel_layout_mode='center_corners', panel_layouts=layouts, episode_schedule=schedule)


@pytest.mark.parametrize('repeats', [1, 2])
def test_complete_floor_layout_repeat_schedule(repeats):
    manifest = manifest_fixture(repeats=repeats)
    schedule = episode_schedule(manifest)
    assert len(schedule) == 60 * repeats
    assert len({(r['floor'], r['repeat'], r['panel_layout_name']) for r in schedule}) == 60 * repeats
    assert len({r['seed_episode_index'] for r in schedule}) == len(schedule)


def test_four_workers_keep_canonical_seeds_without_losing_conditions():
    full = episode_schedule(manifest_fixture())
    actual = []
    for start in range(24, 36, 3):
        worker = episode_schedule(manifest_fixture(range(start, start + 3)))
        assert len(worker) == 15
        actual.extend(worker)
    key = lambda r: (r['repeat'], r['panel_layout_name'], r['floor'], r['seed_episode_index'])
    assert sorted(map(key, actual)) == sorted(map(key, full))


@pytest.mark.parametrize('change', ['layout_offset', 'layout_name', 'lost_episode', 'reordered_episode', 'lost_repeat', 'duplicate_floor', 'wrong_seed'])
def test_tampered_manifest_condition_coverage_is_rejected(change):
    manifest = manifest_fixture(repeats=2)
    if change == 'layout_offset':
        manifest['panel_layouts'][1]['panel_offset_x_m'] = 0.
    elif change == 'layout_name':
        manifest['panel_layouts'][2]['panel_layout_name'] = 'xmin_ymin'
    elif change == 'lost_episode':
        manifest['episode_schedule'].pop()
    elif change == 'reordered_episode':
        manifest['episode_schedule'][0], manifest['episode_schedule'][1] = manifest['episode_schedule'][1], manifest['episode_schedule'][0]
    elif change == 'lost_repeat':
        manifest['episode_schedule'][60]['repeat'] = 0
    elif change == 'duplicate_floor':
        manifest['arguments']['floors'].append(24)
    else:
        manifest['episode_schedule'][-1]['seed_episode_index'] -= 12
    with pytest.raises(ValueError):
        episode_schedule(manifest)


@pytest.mark.parametrize('index', [0, 9, 12, 59, 73, 119])
def test_inference_request_seeds_include_layout_and_repeat(index):
    metadata = episode_schedule(manifest_fixture(repeats=2))[index]
    requests = [dict(chunk_index=k, seed=99 + metadata['seed_episode_index'] * 10000 + k) for k in range(4)]
    assert request_seed_evidence(metadata, requests, 99, layout_count=5) == 99
    requests[-1]['seed'] += 1
    with pytest.raises(ValueError, match='Request inference seed'):
        request_seed_evidence(metadata, requests, 99, layout_count=5)


def geometry_fixture(origin=(0., 6., 0.), x=.01, y=-.025):
    # A simple calibrated camera looks straight along +X. Hand-derived
    # projection: u=320-500*y/depth, v=240-500*(z-1.05)/depth.
    origin = np.asarray(origin)
    cfg = dict(robot_base_x=-.24, robot_base_y=0., table_height=.76,
               button_face_x=.46, button_column_y=.045, button_bottom_z=.98, button_pitch_z=.035,
               global_camera_eye=[-.3, 0., 1.05], global_camera_target=[.46, 0., 1.05],
               panel_randomization=dict(enabled=True, camera_margin_px=24))
    rests = {str(f): (origin + [.463+x, (.045 if f<30 else -.045)+y, .98+(f-24)%6*.035]).tolist() for f in range(24, 36)}
    bounds = np.array([[.459775+x, y-.1, .91], [.509+x, y+.1, 1.225]]) + origin
    wall = np.array([[.52+x, -1.2, 0.], [.57+x, 1.2, 2.4]]) + origin
    corners = np.array(list(product(*zip(bounds[0], bounds[1])))) - origin
    depths = corners[:, 0] + .3
    pixels = np.column_stack((320 - 500*corners[:, 1]/depths, 240 - 500*(corners[:, 2]-1.05)/depths))
    calibration = dict(K=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
                       sensor=dict(world_optical_transform=[[0, 0, -1, -.3], [-1, 0, 0, 0], [0, 1, 0, 1.05], [0, 0, 0, 1]]))
    layout = dict(panel_offset_x_m=x, panel_offset_y_m=y, fixed_rig_unchanged=True,
                  button_rest_positions_world_m=rests, max_button_rest_error_m=0., max_spring_anchor_error_m=0., rest_tolerance_m=.0002,
                  panel_world_bounds_m=bounds.tolist(), back_wall_world_bounds_m=wall.tolist(), back_wall_alignment_error_m=0.,
                  global_camera=dict(resolution=[640,480], coverage_kind='frustum_only', all_inside=True, margin_px=24,
                                     corners_px=pixels.tolist(), corner_depths_m=depths.tolist()))
    metadata = dict(config=cfg, floor=32, robot_base_world_m=(origin+[-.24,0,.76]).tolist(), env_offset_m=origin.tolist(),
                    panel_layout_index=3, panel_layout_name='xmax_ymin', panel_offset_x_m=x, panel_offset_y_m=y, panel_layout=layout)
    return metadata, {'global': calibration}


@pytest.mark.parametrize('origin', [(0., 0., 0.), (0., 6., 0.), (1.5, -12., .3)])
def test_independent_shifted_geometry_keeps_robot_base_frame(origin):
    metadata, collection = geometry_fixture(origin)
    proof = panel_episode_geometry(metadata, collection)
    np.testing.assert_allclose(proof['target_face_base_m'], [.71, -.07, .29], atol=1e-14)
    np.testing.assert_allclose(proof['target_face_world_m'], np.array(origin)+[.47,-.07,1.05], atol=1e-14)
    np.testing.assert_allclose(proof['environment_origin_m'], origin, atol=1e-14)
    assert proof['measured_layout_verified']
    assert proof['minimum_camera_margin_px'] > 24
    assert proof['max_camera_projection_error_px'] < 1e-10


@pytest.mark.parametrize('change', ['missing_x', 'missing_layout', 'body_position', 'stale_offset', 'move_base', 'bounds', 'wall', 'camera_margin', 'camera_pixels', 'camera_depth', 'camera_eye', 'no_collection'])
def test_measured_geometry_detects_stale_hidden_shift_and_projection_errors(change):
    metadata, collection = geometry_fixture()
    layout = metadata['panel_layout']
    if change == 'missing_x':
        metadata.pop('panel_offset_x_m')
    elif change == 'missing_layout':
        metadata.pop('panel_layout')
    elif change == 'body_position':
        layout['button_rest_positions_world_m']['32'][0] -= .01
    elif change == 'stale_offset':
        metadata['panel_offset_x_m'] = 0.
    elif change == 'move_base':
        metadata['robot_base_world_m'][1] += .025
    elif change == 'bounds':
        layout['panel_world_bounds_m'][0][1] += .005
    elif change == 'wall':
        layout['back_wall_world_bounds_m'][0][0] -= .01
    elif change == 'camera_margin':
        layout['global_camera']['corners_px'][0][0] = 23.99
    elif change == 'camera_pixels':
        layout['global_camera']['corners_px'][0][0] += .1
    elif change == 'camera_depth':
        layout['global_camera']['corner_depths_m'][0] += .001
    elif change == 'camera_eye':
        collection['global']['sensor']['world_optical_transform'][0][3] += .01
    else:
        collection = None
    with pytest.raises(ValueError):
        panel_episode_geometry(metadata, collection)


def test_old_unshifted_geometry_and_seed_remain_supported():
    metadata, _ = geometry_fixture(x=0., y=0.)
    for key in ('panel_layout_index', 'panel_layout_name', 'panel_offset_x_m', 'panel_offset_y_m', 'panel_layout', 'env_offset_m'):
        metadata.pop(key)
    metadata['config'].pop('panel_randomization')
    proof = panel_episode_geometry(metadata)
    assert proof['measured_layout_verified'] is False
    assert proof['panel_layout_name'] == 'fixed'
    np.testing.assert_allclose(proof['target_face_base_m'], [.7,-.045,.29])
    manifest = dict(config=metadata['config'], arguments=dict(floors=[24,33,35], episodes_per_floor=2))
    schedule = episode_schedule(manifest)
    assert [r['seed_episode_index'] for r in schedule] == [0,9,11,12,21,23]


def test_aggregate_all_sixty_episodes_instead_of_overwriting_repeated_floors():
    rows = episode_schedule(manifest_fixture())
    for row in rows:
        row.update(success=row['panel_layout_index'] == 0, task_success=row['panel_layout_index'] == 0,
                   termination='target_pressed' if row['panel_layout_index'] == 0 else 'time_limit')
    stats = outcome_statistics(rows)
    assert stats['overall']['episodes'] == 60
    assert stats['overall']['successes'] == 12
    assert stats['overall']['success_rate'] == .2
    assert stats['overall']['termination_counts'] == {'target_pressed':12,'time_limit':48}
    assert all(s['episodes']==5 and s['successes']==1 for s in stats['by_floor'].values())
    assert stats['by_position']['center']['successes'] == 12
    assert all(stats['by_position'][name]['episodes']==12 for name,_,_ in POSITIONS)
    assert all(v['episodes']==1 for floor in stats['by_floor_position'].values() for v in floor.values())
    with pytest.raises(ValueError, match='Duplicate'):
        outcome_statistics(rows+[deepcopy(rows[0])])
