"""CPU-only camera refinement regression, using independent synthetic landmarks."""
import copy
import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('wrist_refinement', ROOT / 'scripts/refine_playground_wrist.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_contour_preserves_concave_mesh_boundary():
    # A convex hull would incorrectly put the right edge at x=4 at y=2.
    triangles = np.array([
        [[0, 0], [4, 0], [2, 2]],
        [[0, 0], [2, 2], [0, 4]],
        [[0, 4], [2, 2], [4, 4]],
    ], dtype=float)
    assert module.horizontal_edges(triangles.reshape(-1, 2), 2).tolist() == [0., 2.]


@pytest.mark.parametrize('bad', [
    {'width_m': float('nan')}, {'width_m': -.01}, {'width_m': .08},
    {'split': 'test'}, {'link': '../link6'}, {'tip_v': 480},
    {'rows': [{'v': 400, 'left_u': 500, 'right_u': 300}]},
])
def test_rejects_invalid_measurements(bad):
    item = {'name': 'synthetic', 'width_m': .05, 'split': 'train', 'tip_v': 351}
    item.update(bad)
    with pytest.raises(ValueError):
        module.validate_edges([item], [640, 480])


def test_recovers_known_camera_and_changes_only_wrist_transform(tmp_path):
    config_path = ROOT / 'configs/calibration/playground_20261107_before_wrist_refine.json'
    cfg = json.loads(config_path.read_text())
    # Real configuration is only a reference geometry; every target below is synthetic.
    camera = cfg['wrist_camera']
    K = np.array([[camera['fx'], 0, camera['cx']], [0, camera['fy'], camera['cy']], [0, 0, 1.]])
    prior = np.array(camera['link6_from_camera_usd']) @ module.D
    delta = np.array([*np.deg2rad([.30, -.15, .08]), .001, -.0008, .0003])
    truth = module.correction(delta, prior)
    original_observations, _ = module.load_observations(
        ROOT / 'configs/calibration/playground_20261107_retry08_video_annotations_v04.json', cfg)
    cubes = {c['name']: c for c in cfg['calibration_objects']}
    observations = []
    for original in original_observations:
        if original['camera'] != 'wrist' or not original['name'].startswith('pregrasp') or not original.get('cube_corners'):
            continue
        observed = {k: copy.deepcopy(original[k]) for k in ['name', 'camera', 'q_rad_m', 'split']}
        observed['cube_corners'] = {}
        for name in original['cube_corners']:
            c = cubes[name]
            xyz = module.C*c['size'][0] @ Rotation.from_euler('z', c['yaw_deg'], degrees=True).as_matrix().T
            xyz += [*c['center_xy'], -cfg['robot_base_height_above_table_m']+c['size'][0]]
            observed['cube_corners'][name] = module.project(K, np.array(original['base_from_link6']) @ truth, xyz).tolist()
        observations.append(observed)
    annotations = tmp_path / 'annotations.json'
    annotations.write_text(json.dumps({'observations': observations}))
    edges = []
    for index, width in enumerate([.035, .0401, .05, .0594]):
        uv = module.project(K, truth, module.finger_triangles(width, 'link8'))
        rows = []
        for v in [375, 395, 415, 435, 455, 475]:
            left, right = module.horizontal_edges(uv, v)
            rows.append({'v': v, 'left_u': float(left), 'right_u': float(right), 'sigma_px': 1.})
        edges.append({'name': f'width{index}', 'width_m': width,
                      'split': 'holdout' if index == 1 else 'train',
                      'tip_v': float(uv[:, 1].min()), 'tip_sigma_px': 1., 'rows': rows})
    edge_path = tmp_path / 'edges.json'
    edge_path.write_text(json.dumps({'observations': edges}))
    candidate, report = module.run(config_path, annotations, edge_path, tmp_path / 'fit',
                                   cube_sigma=1., prior_deg=20., prior_mm=30.)
    inferred = np.array(report['parameters'])
    np.testing.assert_allclose(inferred[:3], delta[:3], atol=np.deg2rad(.01))
    np.testing.assert_allclose(inferred[3:], delta[3:], atol=2e-5)
    for split in ['train', 'holdout']:
        assert all(v['rmse_px'] < .05 for v in report['after'][split]['groups'].values())
    transformed = candidate['wrist_camera'].pop('link6_from_camera_usd')
    unchanged = copy.deepcopy(cfg)
    unchanged['wrist_camera'].pop('link6_from_camera_usd')
    expected_sha = hashlib.sha256(json.dumps(transformed).encode()).hexdigest()
    assert candidate['reconstruction']['wrist_extrinsics_sha256'] == expected_sha
    candidate['reconstruction'].pop('wrist_extrinsics_sha256')
    unchanged['reconstruction'].pop('wrist_extrinsics_sha256')
    assert candidate == unchanged
    assert transformed != cfg['wrist_camera']['link6_from_camera_usd']
    assert json.loads(config_path.read_text()) == cfg


def test_run_refuses_overwriting_existing_output(tmp_path):
    with pytest.raises(FileExistsError):
        module.run(tmp_path/'unused.json', tmp_path/'unused.json',
                   tmp_path/'unused.json', tmp_path)
