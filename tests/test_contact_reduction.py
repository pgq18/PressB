"""Preserve native contact predicates while removing tiny-array overhead."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from pressb.runtime.physics import ContactForceReducer, ContactPathClassifier

DT = 1/120
ANGLE = 20.
TOLERANCE = .003


class NativeFloat3:
    """The sequence interface documented by carb.Float3, without Isaac import."""
    def __init__(self, xyz):
        self.values = tuple(float(x) for x in xyz)

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return self.values[index]


def contact(impulse=(0., 0., .001), normal=(0., 0., 1.), separation=0.):
    return SimpleNamespace(impulse=NativeFloat3(impulse), normal=NativeFloat3(normal), separation=separation)


def reference(points, dt=DT, angle=ANGLE, tolerance=TOLERANCE):
    force, vertical = 0., 0.
    for point in points:
        impulse = np.asarray(point.impulse, dtype=float)
        normal = np.asarray(point.normal, dtype=float)
        separation = float(point.separation)
        if (impulse.shape != (3,) or normal.shape != (3,) or not np.isfinite(impulse).all()
                or not np.isfinite(normal).all() or not math.isfinite(separation)):
            raise RuntimeError('Malformed measured contact data')
        force += float(np.linalg.norm(impulse))/dt
        norm = float(np.linalg.norm(normal))
        if (norm > 1e-8 and abs(float(normal[2]))/norm >= math.cos(math.radians(angle))
                and abs(separation) <= tolerance):
            vertical += abs(float(impulse[2]))/dt
    return force, vertical


def test_force_and_support_sums_match_original_bit_for_bit_on_native_float32_payloads():
    rng = np.random.default_rng(20261113)
    reducer = ContactForceReducer(DT, ANGLE, TOLERANCE)
    for _ in range(256):
        impulses = rng.normal(0., .01, size=(8, 3)).astype(np.float32)
        normals = rng.normal(0., 1., size=(8, 3)).astype(np.float32)
        separations = rng.uniform(-.004, .004, size=8).astype(np.float32)
        points = [contact(i, n, float(s)) for i, n, s in zip(impulses, normals, separations)]
        assert reducer(points) == reference(points)


@pytest.mark.parametrize('normal', [
    (0., 0., 0.),
    (0., 0., float(np.nextafter(1e-8, 0.))),
    (0., 0., 1e-8),
    (0., 0., float(np.nextafter(1e-8, np.inf))),
    (math.sin(math.radians(20.)), 0., math.cos(math.radians(20.))),
    (math.sin(math.radians(20.)), 0., float(np.nextafter(math.cos(math.radians(20.)), 0.))),
    (math.sin(math.radians(20.)), 0., float(np.nextafter(math.cos(math.radians(20.)), np.inf))),
    (math.sin(math.radians(20.)), 0., -math.cos(math.radians(20.))),
    (1e200, 0., 1e200),
    (1e-200, 0., 1e-200),
])
@pytest.mark.parametrize('separation', [
    0., TOLERANCE, -TOLERANCE,
    float(np.nextafter(TOLERANCE, 0.)), float(np.nextafter(TOLERANCE, np.inf)),
    float(np.nextafter(-TOLERANCE, 0.)), float(np.nextafter(-TOLERANCE, -np.inf)),
])
def test_support_decision_boundaries_match_original(normal, separation):
    points = [contact(normal=normal, separation=separation)]
    with np.errstate(over='ignore', under='ignore'):
        assert ContactForceReducer(DT, ANGLE, TOLERANCE)(points) == reference(points)


@pytest.mark.parametrize('force', [float(np.nextafter(.02, 0.)), .02, float(np.nextafter(.02, np.inf))])
def test_grasp_threshold_preserves_original_per_point_and_accumulated_force(force):
    reducer = ContactForceReducer(DT, ANGLE, TOLERANCE)
    points = [contact((force*DT, 0., 0.))]
    assert reducer(points) == reference(points)
    # Multiple headers reach the same threshold: no changed norm or sum order.
    points = [contact((force*DT*.4, 0., 0.)), contact((force*DT*.6, 0., 0.))]
    observed = sum(reducer([p])[0] for p in points)
    expected = sum(reference([p])[0] for p in points)
    assert observed == expected and (observed >= .02) == (expected >= .02)


@pytest.mark.parametrize('field,value', [
    ('impulse', [float('nan'), 0., 0.]), ('impulse', [0., float('inf'), 0.]),
    ('normal', [float('nan'), 0., 0.]), ('normal', [0., 0., float('-inf')]),
    ('separation', float('nan')), ('separation', float('inf')),
    ('impulse', [1., 2.]), ('normal', [1., 2.]),
    ('normal', [[1.], [2.], [3.]]), ('normal', np.zeros((3, 1))),
    ('normal', tuple(np.array([value]) for value in (1., 2., 3.))),
    ('impulse', np.zeros((3, 1))),
])
def test_invalid_shapes_and_nonfinite_contacts_are_still_rejected(field, value):
    point = contact()
    setattr(point, field, value)
    reducer = ContactForceReducer(DT, ANGLE, TOLERANCE)
    with pytest.raises(RuntimeError, match='contact'):
        reducer([point])
    with pytest.raises(RuntimeError):
        reference([point])


def test_fixed_shape_norm_skips_dispatch_and_normal_boundary_falls_back(monkeypatch):
    original = np.linalg.norm
    calls = []
    def measured(array):
        calls.append(np.array(array))
        return original(array)
    monkeypatch.setattr(np.linalg, 'norm', measured)
    reducer = ContactForceReducer(DT, ANGLE, TOLERANCE)
    reducer([contact(normal=(0., 0., 1.))])
    assert len(calls) == 0  # Exact impulse dot/sqrt with no generic dispatch.
    calls.clear()
    reducer([contact(normal=(0., 0., 1e-8))])
    assert len(calls) == 1
    calls.clear()
    reducer([contact(normal=(math.sin(math.radians(20.)), 0., math.cos(math.radians(20.))))])
    assert len(calls) == 1


def test_cached_topology_keeps_exact_root_and_collider_descendant_matching():
    roots = ('/World/Red', '/World/Blue', '/World/Piper/link7', '/World/Top')
    classify = ContactPathClassifier(roots)
    paths = ['/Unknown0', '/Unknown1', '/World/Red/Geometry/part', '/World/Piper/link7/collider']
    assert classify(paths) == (True, False, True, False)
    assert classify(paths) == (True, False, True, False)
    paths[2] = '/World/Red_other/Geometry/part'
    assert classify(paths) == (False, False, True, False)
    # The same roots named in both actor and collider still produce one flag.
    assert classify(['/World/Red', '/World/Top', '/World/Red/shape', '/World/Top/shape']) == (True, False, False, True)


@pytest.mark.parametrize('layout', ['contiguous', 'strided', 'reverse'])
def test_validated_float64_norm_keeps_numpy_dot_sqrt_order_for_all_memory_layouts(layout):
    rng = np.random.default_rng(731)
    reducer = ContactForceReducer(DT, ANGLE, TOLERANCE)
    for scale in (0., 1e-200, 1e-10, 1., 1e100, 1e200):
        for _ in range(32):
            storage = rng.normal(size=6)*scale
            impulse = (storage[:3].copy() if layout == 'contiguous' else
                       storage[::2] if layout == 'strided' else storage[2::-1])
            before = impulse.copy()
            point = contact()
            point.impulse = impulse
            with np.errstate(over='ignore', under='ignore', invalid='ignore'):
                assert reducer([point]) == reference([point])
            np.testing.assert_array_equal(impulse, before)
