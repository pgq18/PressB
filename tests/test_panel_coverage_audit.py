"""Coverage evidence cannot conceal wrong cells, geometry or missing corners."""
from copy import deepcopy
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_panel_coverage import coverage_counts, inspect_metadata, offset_cell, project_bounds


def fixture(flat=45, floor=24):
    settings = dict(enabled=True, sampling_mode='stratified_grid', grid_shape=[10, 10], grid_seed=20260930,
        boundary_mode='corners', min_offset_x_m=-.01, max_offset_x_m=.01, min_offset_y_m=-.025,
        max_offset_y_m=.025, camera_margin_px=24, max_attempts=8)
    cfg = dict(panel_randomization=settings, button_face_x=.46, button_column_y=.045, button_bottom_z=.98,
        button_pitch_z=.035, robot_base_x=-.24, robot_base_y=0., table_height=.76,
        global_camera_eye=[-1., 0., 1.], global_camera_target=[.46, 0., 1.])
    camera = [[0., 0., -1., -1.], [-1., 0., 0., 0.], [0., 1., 0., 1.], [0., 0., 0., 1.]]
    collection = dict(raw_schema_version=11, fps=30, physics_hz=120, capture_stride=4, action_horizon_s=1/30,
        seed=100, collection_fingerprint='test', config=cfg,
        **{'global': dict(width=640, height=480, K=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
                         sensor=dict(world_optical_transform=camera))})
    stream = [settings['grid_seed'], floor, 0x47524944]
    ordinal = int(np.flatnonzero(np.random.default_rng(np.random.SeedSequence(stream)).permutation(100) == flat)[0])
    eid = ordinal * 12 + floor - 24
    cell = [flat // 10, flat % 10]
    bounds = {axis: np.linspace(settings[f'min_offset_{axis}_m'], settings[f'max_offset_{axis}_m'], 11)[i:i+2].tolist()
              for axis, i in zip('xy', cell)}
    corner = all(i in (0, 9) for i in cell)
    xy = [settings[f'{"min" if i == 0 else "max"}_offset_{axis}_m'] if corner else float(np.mean(bounds[axis]))
          for axis, i in zip('xy', cell)]
    x, y = xy
    trajectory = dict(physics_samples_checked=20000, maximum_fk_target_error_m=.00001,
        conservative_clearances=dict(table_m=.05, wall_or_panel_m=.11, fixed_camera_m=.016))
    validation = dict(panel_offset_x_m=x, panel_offset_y_m=y, all_buttons_reachable=True,
        reachable_floors=list(range(24,36)), trajectory_validation=trajectory,
        camera_coverage=dict(all_inside=True, required_margin_px=24., minimum_margin_px=100.))
    proof = dict(**settings, axis='world_xy', distribution='stratified_grid_with_safety_rejection',
        grid_permutation_seed_stream=stream, grid_permutation_order='flat_x_major_y_minor', episode_index=ordinal,
        grid_flat_index=flat, grid_cell_index_xy=cell, grid_cell_bounds_m=bounds, boundary_corner=corner,
        seed_stream=[100+eid, floor, 0x50414E45], panel_offset_x_m=x, panel_offset_y_m=y,
        attempts=1, rejected_candidates=[], validation=validation)
    env = np.array([0., 2., 0.])
    rests = {str(f): (env + [cfg['button_face_x']+.003+x, (.045 if f < 30 else -.045)+y,
                            .98+((f-24)%6)*.035]).tolist() for f in range(24,36)}
    panel = np.array([[.46+x-.000225,y-.1,.91],[.46+x+.049,y+.1,1.225]])+env
    pixels,depths=project_bounds(panel,collection,env)
    layout = dict(panel_offset_x_m=x, panel_offset_y_m=y, fixed_rig_unchanged=True,
        button_rest_positions_world_m=rests, max_button_rest_error_m=0., max_spring_anchor_error_m=0.,
        panel_world_bounds_m=panel.tolist(), back_wall_world_bounds_m=(np.array([[.52+x,-1.2,0],[.57+x,1.2,2.4]])+env).tolist(),
        back_wall_alignment_error_m=0., global_camera=dict(resolution=[640,480],all_inside=True,
            coverage_kind='frustum_only',margin_px=24.,corners_px=pixels.tolist(),corner_depths_m=depths.tolist()))
    meta = dict(episode_id=eid,floor=floor,task=f'Press {floor} floor.',seed=100+eid,success=True,
        **{k:collection[k] for k in ('raw_schema_version','fps','physics_hz','capture_stride','action_horizon_s','collection_fingerprint')},
        panel_offset_x_m=x,panel_offset_y_m=y,env_offset_m=env.tolist(),robot_base_world_m=(env+[-.24,0,.76]).tolist(),
        variation=dict(panel_offset_x_m=x,panel_offset_y_m=y,panel_randomization=proof,validation=deepcopy(trajectory)),panel_layout=layout)
    return meta,collection


@pytest.mark.parametrize('flat', [0,9,90,99,45])
def test_exact_corners_and_interior_cell_from_actual_geometry(flat):
    meta,collection=fixture(flat)
    result=inspect_metadata(meta,collection)
    assert result['grid_flat_index']==flat
    assert result['boundary_corner']==(flat in (0,9,90,99))
    np.testing.assert_allclose(result['measured_panel_offset_xy_m'],[meta['panel_offset_x_m'],meta['panel_offset_y_m']],atol=1e-15)
    assert result['minimum_actual_camera_margin_px']>24


@pytest.mark.parametrize('change', ['cell','ordinal','offset','rest','wall','projection','range','clearance','corner','rejection'])
def test_invalid_coverage_evidence_is_rejected(change):
    meta,collection=fixture(0 if change=='corner' else 45)
    proof=meta['variation']['panel_randomization']
    if change=='cell':proof['grid_cell_index_xy']=[5,5]
    elif change=='ordinal':proof['episode_index']+=1
    elif change=='offset':meta['panel_offset_y_m']+=.01
    elif change=='rest':meta['panel_layout']['button_rest_positions_world_m']['24'][0]+=.001
    elif change=='wall':meta['panel_layout']['back_wall_world_bounds_m'][0][0]+=.002
    elif change=='projection':meta['panel_layout']['global_camera']['corners_px'][0][0]+=1
    elif change=='range':proof['max_offset_x_m']=.02
    elif change=='clearance':proof['validation']['trajectory_validation']['conservative_clearances']['fixed_camera_m']=.0149
    elif change=='corner':proof['boundary_corner']=False
    else:
        proof['attempts']=2
        proof['rejected_candidates']=[dict(panel_offset_x_m=.009,panel_offset_y_m=.024,reason='invalid')]
    with pytest.raises(ValueError):inspect_metadata(meta,collection)


def test_full_coverage_needs_one_unique_cell_per_floor_and_four_corners():
    rows=[dict(episode_id=ordinal*12+f-24, floor=f,episode_index=ordinal,
               grid_cell_index_xy=[ordinal//10,ordinal%10],boundary_corner=ordinal in (0,9,90,99))
          for f in range(24,36) for ordinal in range(100)]
    assert all(v['occupied_cells']==100 and v['exact_corners']==4 for v in coverage_counts(rows).values())
    with pytest.raises(ValueError,match='full 100-cell'):coverage_counts(rows[:-1])
    assert coverage_counts(rows[:1],allow_partial=True)['24']['occupied_cells']==1
    duplicate=deepcopy(rows)
    duplicate[1]['grid_cell_index_xy']=duplicate[0]['grid_cell_index_xy']
    with pytest.raises(ValueError,match='repeated grid cells'):coverage_counts(duplicate)


def test_outer_maximum_belongs_to_last_cell_and_outside_is_rejected():
    assert offset_cell(-.01,-.01,.01,10)==0
    assert offset_cell(.01,-.01,.01,10)==9
    with pytest.raises(ValueError):offset_cell(.01000001,-.01,.01,10)
