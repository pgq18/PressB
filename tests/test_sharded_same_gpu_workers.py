"""Prepared coverage for independent collector workers sharing one GPU."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sharded_same_gpu', ROOT/'scripts/run_sharded_manipulation_collection.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def collection(tmp_path):
    raw = tmp_path/'canonical'
    raw.mkdir()
    manifest = dict(task_id='tabletop_cleanup', instruction='Cleanup.', target_episodes=6,
                    seed=10, noise_deg=.12, settle_seconds=1., pair_schedule=[0,1,2,0,1,2],
                    config=dict(gpu=1, num_envs=8, log_dir=str(raw/'simulation')))
    runner.write_json(raw/'manifest.json', manifest)
    return raw, manifest


def test_default_preserves_one_worker_per_gpu_cli_and_plan(tmp_path, collection, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, 'run', lambda args: calls.append(args) or 0)
    assert runner.main(['--canonical', str(tmp_path), '--gpus', '1']) == 0
    assert calls[0].workers_per_gpu == 1
    assert calls[0].gpus == [1]
    assert runner.expand_worker_gpus([0, 1]) == [0, 1]
    raw, manifest = collection
    plan = runner.make_plan(raw, manifest, {}, calls[0].gpus, 8, 5000)
    assert len(plan['workers']) == 1
    assert plan['workers'][0]['episode_indices'] == list(range(6))
    assert plan['workers'][0]['attempt_stride'] == 1
    runner.validate_plan(plan, raw, manifest, {}, [1], 8, 5000)


def test_cli_expands_unique_physical_devices_into_workers(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, 'run', lambda args: calls.append(args) or 0)
    assert runner.main(['--canonical', str(tmp_path), '--gpus', '1', '--workers-per-gpu', '2']) == 0
    assert calls[0].gpus == [1, 1]
    assert calls[0].workers_per_gpu == 2
    assert runner.expand_worker_gpus([1, 0], 2) == [1, 1, 0, 0]


def test_same_gpu_workers_have_disjoint_episode_attempt_and_command_ownership(collection):
    raw, manifest = collection
    runner.write_json(raw/'progress.json', {'attempts':13})
    gpus = runner.expand_worker_gpus([1], 2)
    plan = runner.make_plan(raw, manifest, {}, gpus, 8, 5000)
    first, second = plan['workers']
    assert [w['gpu'] for w in plan['workers']] == [1, 1]
    assert first['episode_indices'] == [0, 2, 4]
    assert second['episode_indices'] == [1, 3, 5]
    assert set(first['episode_indices']).isdisjoint(second['episode_indices'])
    assert set(first['episode_indices']) | set(second['episode_indices']) == set(range(6))
    attempts = [{w['attempt_offset'] + w['attempt_stride'] * i for i in range(1, 101)}
                for w in plan['workers']]
    seeds = [{manifest['seed'] + 1009 * attempt for attempt in values} for values in attempts]
    assert attempts[0].isdisjoint(attempts[1]) and seeds[0].isdisjoint(seeds[1])
    assert min(attempts[0] | attempts[1]) > 13
    assert first['attempt_stride'] == second['attempt_stride'] == 2
    for field in ('output', 'episode_indices_path', 'runtime_metadata_path', 'receipt', 'log'):
        assert first[field] != second[field]
    commands = [runner.worker_command(w, plan, manifest, Path('/python')) for w in plan['workers']]
    for worker, command in zip(plan['workers'], commands):
        for flag, expected in (('--gpu', '1'), ('--num-envs', '8'), ('--attempt-stride', '2'),
                               ('--output', worker['output']), ('--episode-indices', worker['episode_indices_path']),
                               ('--attempt-offset', str(worker['attempt_offset']))):
            assert command[command.index(flag)+1] == expected
    assert commands[0] != commands[1]


def test_same_gpu_plan_resumes_only_with_identical_worker_configuration(collection):
    raw, manifest = collection
    gpus = runner.expand_worker_gpus([1], 2)
    plan = runner.make_plan(raw, manifest, {}, gpus, 8, 5000)
    path = raw/'collection_shards.json'
    runner.write_json(path, plan)
    before = path.read_bytes()
    saved = runner.read_json(path)
    runner.validate_plan(saved, raw, manifest, {}, runner.expand_worker_gpus([1], 2), 8, 5000)
    assert path.read_bytes() == before
    for changed, num_envs, limit in (([1], 8, 5000), ([2, 2], 8, 5000),
            ([1, 1, 1], 8, 5000), ([1, 1], 16, 5000), ([1, 1], 8, 5001)):
        with pytest.raises(ValueError, match='configuration'):
            runner.validate_plan(saved, raw, manifest, {}, changed, num_envs, limit)
    assert path.read_bytes() == before


@pytest.mark.parametrize('value', [0, -1, True, False, 1.5, '2'])
def test_workers_per_gpu_rejects_nonpositive_or_noninteger_values(value):
    with pytest.raises(ValueError, match='positive integer'):
        runner.expand_worker_gpus([1], value)


@pytest.mark.parametrize('gpus', [[], [1, 1], [-1], [True], [False], [1.0], ['1']])
def test_physical_gpu_selection_is_validated_before_expansion(gpus):
    with pytest.raises(ValueError, match='distinct, nonnegative'):
        runner.expand_worker_gpus(gpus, 2)


@pytest.mark.parametrize('options', [
    ['--workers-per-gpu', '0'], ['--workers-per-gpu', '-1'],
    ['--workers-per-gpu', 'True'], ['--workers-per-gpu', 'False'],
    ['--gpus', '1', '1', '--workers-per-gpu', '2'], ['--gpus', '-1'],
])
def test_invalid_cli_values_fail_before_run(tmp_path, monkeypatch, options):
    calls = []
    monkeypatch.setattr(runner, 'run', lambda args: calls.append(args) or 0)
    with pytest.raises(SystemExit) as error:
        runner.main(['--canonical', str(tmp_path), *options])
    assert error.value.code == 2
    assert not calls
