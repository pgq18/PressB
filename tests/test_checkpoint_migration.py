"""Offline revision migration preserves CPU learner state and strict resume guards."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pressb.runtime.checkpoint_migration import migrate_checkpoint
from pressb.runtime.contract import fingerprint
from pressb.runtime.trainer import FrozenDSRLConfig, GenericTrainer, _schema_fingerprint
from test_rl_nodes_trainer import Inference, Simulation, make_runner


@pytest.fixture(autouse=True)
def one_cpu_thread():
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prior)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, sort_keys=True, indent=2))
    return path


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _contract_fingerprint(identity):
    contract = identity["contract"]
    contract["fingerprint"] = fingerprint({key: value for key, value in contract.items()
                                            if key not in ("num_envs", "fingerprint")})


def _next_identity(identity):
    result = deepcopy(identity)
    for kind in ("inference", "simulation"):
        result[kind]["source_revision"] = f"new-{kind}-revision"
        result["contract"]["identity"][kind]["source_revision"] = result[kind]["source_revision"]
    _contract_fingerprint(result)
    return result


def _assert_exact(left, right, path="payload"):
    """Compare serialized values, including tensor/array dtype and raw bytes."""
    assert type(left) is type(right), path
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape, path
        assert left.cpu().contiguous().numpy().tobytes() == right.cpu().contiguous().numpy().tobytes(), path
    elif isinstance(left, np.ndarray):
        assert left.dtype == right.dtype and left.shape == right.shape, path
        assert left.tobytes() == right.tobytes(), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            _assert_exact(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_exact(a, b, f"{path}[{index}]")
    else:
        assert left == right, path


def _assert_payload_preserved(source, migrated, target_identity, *, frozen=None):
    expected = deepcopy(source)
    expected["extra"]["schema_fingerprint"] = _schema_fingerprint(target_identity)
    expected["extra"]["contract"] = deepcopy(target_identity["contract"])
    if frozen is not None:
        expected["extra"]["frozen_dsrl"].update(
            checkpoint_sha256=_sha(frozen),
            schema_fingerprint=target_identity["frozen_dsrl"]["schema_fingerprint"],
            checkpoint=str(Path(frozen).resolve()),
        )
    actual = deepcopy(migrated)
    assert isinstance(actual["extra"].pop("checkpoint_migration"), dict)
    _assert_exact(expected, actual)


def _resume(case, checkpoint, output, *, frozen=None):
    cfg = deepcopy(case.runner.config)
    cfg.output_dir = str(output)
    cfg.checkpoint.resume = str(checkpoint)
    cfg.rollout.max_transitions = 8
    if frozen is not None:
        cfg.frozen_dsrl.checkpoint = str(frozen)
        cfg.frozen_dsrl.checkpoint_sha256 = _sha(frozen)
    inference = deepcopy(case.inference.meta)
    simulation = deepcopy(case.simulation.meta)
    inference["source_revision"] = case.target["inference"]["source_revision"]
    simulation["source_revision"] = case.target["simulation"]["source_revision"]
    return GenericTrainer(cfg, inference_client=Inference(inference), simulation_client=Simulation(simulation))


@pytest.fixture
def revision_case(tmp_path):
    runner, inference, simulation = make_runner(tmp_path / "source", geometry=(4, 3, 3, 3), steps=4)
    inference.meta["source_revision"] = "old-inference-revision"
    simulation.meta["source_revision"] = "old-simulation-revision"
    checkpoint = Path(runner.run()["last_checkpoint"])
    payload = _load(checkpoint)
    # Future or extension-owned state must survive without a migration allowlist.
    payload["extension_state"] = (b"opaque", torch.tensor([1, 2], dtype=torch.int64))
    payload["extra"]["extension_state"] = {"array": np.array([1., -0.], dtype=np.float64), "notes": [None, True]}
    torch.save(payload, checkpoint)
    source_path = runner.output / "negotiated.json"
    source = json.loads(source_path.read_text())
    target = _next_identity(source)
    return SimpleNamespace(runner=runner, inference=inference, simulation=simulation,
                           checkpoint=checkpoint, source=source, target=target, source_path=source_path,
                           target_path=_json(tmp_path / "target_identity.json", target),
                           regression=_json(tmp_path / "regression.json", {"cpu": {"passed": 1, "failed": 0}}),
                           output=tmp_path / "migrated.pt")


def _migrate(case, **kwargs):
    options = dict(allow_source_revision_change=True, regression_report=case.regression)
    options.update(kwargs)
    return migrate_checkpoint(case.checkpoint, case.source_path, case.target_path, case.output, **options)


def test_revision_migration_preserves_all_state_and_enables_strict_resume(revision_case, tmp_path):
    case = revision_case
    original_sha = _sha(case.checkpoint)
    original = _load(case.checkpoint)
    assert original["actor_optimizer"]["state"] and original["extra"]["replay"]["size"] == 4
    report = _migrate(case)
    migrated = _load(case.output)
    _assert_payload_preserved(original, migrated, case.target)
    assert _sha(case.checkpoint) == original_sha
    assert case.source["contract"]["fingerprint"] != case.target["contract"]["fingerprint"]
    assert isinstance(report, dict)
    sidecar = Path(str(case.output) + ".migration.json")
    assert json.loads(sidecar.read_text()) == report
    provenance = migrated["extra"]["checkpoint_migration"]
    assert provenance["format_version"] == 1 and provenance["kind"] == "source_revision_only"
    assert provenance["allow_source_revision_change"] is True
    assert provenance["source_identity"] == case.source
    assert provenance["target_identity"] == case.target
    assert provenance["source_schema_fingerprint"] == _schema_fingerprint(case.source)
    assert provenance["target_schema_fingerprint"] == _schema_fingerprint(case.target)
    assert provenance["limitation"]
    for key, path in (("source_checkpoint", case.checkpoint), ("source_identity_file", case.source_path),
                      ("target_identity_file", case.target_path), ("regression_report", case.regression)):
        assert provenance[key] == {"path": str(path.resolve()), "sha256": _sha(path), "bytes": path.stat().st_size}
    assert report == {**provenance, "output_checkpoint": {"path": str(case.output.resolve()),
                                                         "sha256": _sha(case.output),
                                                         "bytes": case.output.stat().st_size},
                      "report_path": str(sidecar.resolve())}
    changed_paths = {tuple(change["path"]) for change in provenance["differences"]}
    assert ("inference", "source_revision") in changed_paths
    assert ("simulation", "source_revision") in changed_paths

    old = _resume(case, case.checkpoint, tmp_path / "old-rejected")
    with pytest.raises(ValueError, match="fingerprint"):
        old.run()
    assert old.simulation.resets == []
    resumed = _resume(case, case.output, tmp_path / "resumed")
    result = resumed.run()
    assert result["state"] == "completed" and result["transitions"] == 8
    assert result["updates"] > original["updates"]
    for name, value in original["extra"]["replay"]["arrays"].items():
        np.testing.assert_array_equal(resumed.replay.arrays[name][:4], value)


@pytest.mark.parametrize("approved", [False, True])
def test_cli_requires_opt_in_and_writes_a_reviewable_derived_copy(revision_case, approved):
    case = revision_case
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "scripts/migrate_rl_checkpoint.py"),
               "--source-checkpoint", str(case.checkpoint),
               "--source-identity", str(case.source_path),
               "--target-identity", str(case.target_path),
               "--output", str(case.output), "--regression-report", str(case.regression)]
    if approved:
        command.append("--allow-source-revision-change")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=30)
    if not approved:
        assert result.returncode == 2 and "--allow-source-revision-change" in result.stderr
        assert not case.output.exists()
    else:
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        assert report["output_checkpoint"]["sha256"] == _sha(case.output)
        assert json.loads(Path(report["report_path"]).read_text()) == report
        _assert_payload_preserved(_load(case.checkpoint), _load(case.output), case.target)


def test_migration_does_not_restore_or_consume_process_rng_state(revision_case):
    before = {"python": random.getstate(), "numpy": np.random.get_state(),
              "torch": torch.get_rng_state().clone()}
    _migrate(revision_case)
    after = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    _assert_exact(before, after)


@pytest.mark.parametrize("interrupted_link", [1, 2])
def test_publication_interrupt_cleans_both_own_files_and_allows_retry(revision_case, monkeypatch, interrupted_link):
    from pressb.runtime import checkpoint_migration

    case = revision_case
    original_sha = _sha(case.checkpoint)
    original_link = checkpoint_migration.os.link
    links = []

    def interrupt_after_link(source, destination):
        original_link(source, destination)
        links.append(destination)
        if len(links) == interrupted_link:
            raise KeyboardInterrupt("injected publication interrupt")

    with monkeypatch.context() as patch:
        patch.setattr(checkpoint_migration.os, "link", interrupt_after_link)
        with pytest.raises(KeyboardInterrupt):
            _migrate(case)
    assert not case.output.exists()
    assert not Path(str(case.output) + ".migration.json").exists()
    assert not list(case.output.parent.glob(".migrated.pt.*"))
    assert _sha(case.checkpoint) == original_sha
    _migrate(case)
    assert case.output.exists()


@pytest.mark.parametrize("changed_kind", ["inference", "simulation"])
def test_revision_migration_allows_only_one_node_revision_to_change(revision_case, changed_kind):
    case = revision_case
    unchanged_kind = "simulation" if changed_kind == "inference" else "inference"
    case.target[unchanged_kind]["source_revision"] = case.source[unchanged_kind]["source_revision"]
    case.target["contract"]["identity"][unchanged_kind]["source_revision"] = case.source[unchanged_kind]["source_revision"]
    _contract_fingerprint(case.target)
    _json(case.target_path, case.target)
    _migrate(case)
    _assert_payload_preserved(_load(case.checkpoint), _load(case.output), case.target)


@pytest.mark.parametrize("omission", ["flag", "report", "missing_report"])
def test_migration_requires_explicit_opt_in_and_existing_regression_report(revision_case, omission):
    case = revision_case
    options = dict(allow_source_revision_change=True, regression_report=case.regression)
    if omission == "flag":
        options.pop("allow_source_revision_change")
    elif omission == "report":
        options.pop("regression_report")
    else:
        options["regression_report"] = case.regression.with_name("does-not-exist.json")
    with pytest.raises((ValueError, TypeError, FileNotFoundError)):
        migrate_checkpoint(case.checkpoint, case.source_path, case.target_path, case.output, **options)
    assert not case.output.exists()
    assert not Path(str(case.output) + ".migration.json").exists()


@pytest.mark.parametrize("destination", ["source", "existing", "source_symlink", "source_hardlink", "existing_report"])
def test_migration_never_overwrites_source_output_or_report(revision_case, destination):
    case = revision_case
    original_sha = _sha(case.checkpoint)
    if destination == "source":
        case.output = case.checkpoint
    elif destination == "existing":
        case.output.write_bytes(b"existing output")
    elif destination == "source_symlink":
        case.output.symlink_to(case.checkpoint)
    elif destination == "source_hardlink":
        case.output.hardlink_to(case.checkpoint)
    else:
        Path(str(case.output) + ".migration.json").write_text("existing report")
    before = case.output.read_bytes() if case.output.exists() else None
    with pytest.raises((ValueError, FileExistsError)):
        _migrate(case)
    assert _sha(case.checkpoint) == original_sha
    assert (case.output.read_bytes() if case.output.exists() else None) == before
    if destination == "existing_report":
        assert Path(str(case.output) + ".migration.json").read_text() == "existing report"


@pytest.mark.parametrize("field", ["model", "task", "reward", "dimension", "discount", "projection", "weights",
                                   "nested_revision", "nested_capacity"])
def test_migration_rejects_changed_learning_semantics(revision_case, field):
    case = revision_case
    target = case.target
    if field in ("model", "task", "weights"):
        kind, key = {"model": ("inference", "model_id"), "task": ("simulation", "task_id"),
                     "weights": ("inference", "weights_sha256")}[field]
        target[kind][key] = "changed"
        target["contract"]["identity"][kind][key] = "changed"
    elif field == "reward":
        target["simulation"]["reward_schema"] = {"success": 2.}
    elif field == "dimension":
        target["contract"]["action_dim"] += 1
    elif field == "discount":
        target["contract"]["discount"]["gamma"] = .8
    elif field == "nested_revision":
        target["inference"]["identity"] = {"source_revision": "unapproved-nested-revision"}
    elif field == "nested_capacity":
        target["simulation"]["num_envs"] = 64
    else:
        target["action_projection"]["mode"] = "clip"
    _contract_fingerprint(target)
    _json(case.target_path, target)
    with pytest.raises(ValueError):
        _migrate(case)
    assert not case.output.exists()


@pytest.mark.parametrize("side", ["source", "target"])
@pytest.mark.parametrize("invalid", ["top_revision", "mirror_revision", "contract_fingerprint"])
def test_migration_rejects_inconsistent_identity_or_contract_digest(revision_case, side, invalid):
    case = revision_case
    identity = deepcopy(getattr(case, side))
    if invalid == "top_revision":
        identity["inference"]["source_revision"] = "unmirrored"
    elif invalid == "mirror_revision":
        identity["contract"]["identity"]["simulation"]["source_revision"] = "unmirrored"
        _contract_fingerprint(identity)
    else:
        identity["contract"]["fingerprint"] = "0" * 64
    _json(getattr(case, f"{side}_path"), identity)
    with pytest.raises(ValueError):
        _migrate(case)
    assert not case.output.exists()


@pytest.mark.parametrize("side", ["source", "target"])
@pytest.mark.parametrize("invalid", [None, "", 123])
def test_source_revisions_must_exist_as_nonempty_strings(revision_case, side, invalid):
    case = revision_case
    identity = deepcopy(getattr(case, side))
    for mapping in (identity["inference"], identity["contract"]["identity"]["inference"]):
        if invalid is None:
            mapping.pop("source_revision")
        else:
            mapping["source_revision"] = invalid
    _contract_fingerprint(identity)
    _json(getattr(case, f"{side}_path"), identity)
    if side == "source":
        # Preserve the source checkpoint binding so this exercises revision
        # validation instead of an unrelated source-schema mismatch.
        payload = _load(case.checkpoint)
        payload["extra"]["contract"] = identity["contract"]
        payload["extra"]["schema_fingerprint"] = _schema_fingerprint(identity)
        torch.save(payload, case.checkpoint)
    with pytest.raises(ValueError):
        _migrate(case)
    assert not case.output.exists()


@pytest.mark.parametrize("field", ["schema_fingerprint", "contract"])
def test_migration_rejects_checkpoint_not_bound_to_source_identity(revision_case, field):
    case = revision_case
    payload = _load(case.checkpoint)
    if field == "schema_fingerprint":
        payload["extra"][field] = "0" * 64
    else:
        payload["extra"][field]["action_dim"] += 1
    torch.save(payload, case.checkpoint)
    with pytest.raises(ValueError):
        _migrate(case)
    assert not case.output.exists()


def test_migration_allows_capacity_and_nonlearning_contract_fields_only(revision_case, tmp_path):
    case = revision_case
    case.target["contract"].update(num_envs=64, terminal_observation_mode="state_only_if_no_bootstrap",
                                   fused_step_reset=True)
    _contract_fingerprint(case.target)
    _json(case.target_path, case.target)
    original = _load(case.checkpoint)
    _migrate(case)
    migrated = _load(case.output)
    _assert_payload_preserved(original, migrated, case.target)
    assert migrated["extra"]["terminal_observation_mode"] == "full"
    assert migrated["extra"]["fused_step_reset"] is False
    # Negotiating ordinary 2-env/full storage still succeeds because those
    # contract fields are excluded by the existing trainer schema fingerprint.
    assert _resume(case, case.output, tmp_path / "capacity-resumed").run()["transitions"] == 8


@pytest.fixture
def frozen_chain(revision_case, tmp_path):
    noise = revision_case
    _migrate(noise)
    runner, inference, simulation = make_runner(tmp_path / "residual-source", method="action_residual",
                                               geometry=(4, 3, 3, 3), steps=4)
    inference.meta = deepcopy(noise.inference.meta)
    simulation.meta = deepcopy(noise.simulation.meta)
    runner.config.frozen_dsrl = FrozenDSRLConfig(str(noise.checkpoint), deterministic=True)
    checkpoint = Path(runner.run()["last_checkpoint"])
    source_path = runner.output / "negotiated.json"
    source = json.loads(source_path.read_text())
    target = _next_identity(source)
    target["frozen_dsrl"].update(checkpoint_sha256=_sha(noise.output),
                                schema_fingerprint=_schema_fingerprint(noise.target))
    residual = SimpleNamespace(runner=runner, inference=inference, simulation=simulation,
                               checkpoint=checkpoint, source=source, target=target, source_path=source_path,
                               target_path=_json(tmp_path / "residual-target.json", target),
                               regression=noise.regression, output=tmp_path / "residual-migrated.pt")
    return noise, residual


def _migrate_residual(noise, residual, **kwargs):
    options = dict(source_frozen_checkpoint=noise.checkpoint, target_frozen_checkpoint=noise.output)
    options.update(kwargs)
    return _migrate(residual, **options)


def test_frozen_chain_migration_preserves_actor_and_resumes_strictly(frozen_chain, tmp_path):
    noise, residual = frozen_chain
    before = _load(residual.checkpoint)
    original_sha = _sha(residual.checkpoint)
    _migrate_residual(noise, residual)
    _assert_payload_preserved(before, _load(residual.output), residual.target, frozen=noise.output)
    assert _sha(residual.checkpoint) == original_sha
    old = _resume(residual, residual.checkpoint, tmp_path / "old-residual-rejected", frozen=noise.output)
    with pytest.raises(ValueError, match="fingerprint"):
        old.run()
    assert old.simulation.resets == []
    resumed = _resume(residual, residual.output, tmp_path / "residual-resumed", frozen=noise.output)
    result = resumed.run()
    assert result["transitions"] == 8 and result["updates"] > before["updates"]
    assert resumed.frozen_dsrl.verify()["matches_initial"]
    assert resumed.frozen_dsrl.actor_sha256 == before["extra"]["frozen_dsrl"]["actor_sha256"]
    np.testing.assert_array_equal(resumed.replay.arrays["base_action"][:4],
                                  before["extra"]["replay"]["arrays"]["base_action"])


@pytest.mark.parametrize("missing", ["source_frozen_checkpoint", "target_frozen_checkpoint", "both"])
def test_frozen_chain_requires_original_and_derived_noise_checkpoints(frozen_chain, missing):
    noise, residual = frozen_chain
    options = {missing: None} if missing != "both" else dict(source_frozen_checkpoint=None,
                                                            target_frozen_checkpoint=None)
    with pytest.raises(ValueError):
        _migrate_residual(noise, residual, **options)
    assert not residual.output.exists()


@pytest.mark.parametrize("field", ["deterministic", "learner_config"])
def test_frozen_chain_rejects_sampling_or_learner_changes(frozen_chain, field):
    noise, residual = frozen_chain
    if field == "deterministic":
        residual.target["frozen_dsrl"][field] = False
    else:
        residual.target["frozen_dsrl"][field]["noise_scale"] += .1
    _json(residual.target_path, residual.target)
    with pytest.raises(ValueError):
        _migrate_residual(noise, residual)
    assert not residual.output.exists()


@pytest.mark.parametrize("field", ["checkpoint_sha256", "schema_fingerprint"])
def test_frozen_chain_rejects_residual_checkpoint_binding_mismatch(frozen_chain, field):
    noise, residual = frozen_chain
    payload = _load(residual.checkpoint)
    payload["extra"]["frozen_dsrl"][field] = "0" * 64
    torch.save(payload, residual.checkpoint)
    with pytest.raises(ValueError):
        _migrate_residual(noise, residual)
    assert not residual.output.exists()


@pytest.mark.parametrize("field", ["actor", "actor_dtype", "config", "optimizer", "replay", "rng", "provenance"])
def test_frozen_chain_rejects_tampered_derived_noise_even_with_matching_target_sha(frozen_chain, field):
    noise, residual = frozen_chain
    payload = _load(noise.output)
    if field == "actor":
        payload["actor"]["mean.bias"] += .1
    elif field == "actor_dtype":
        payload["actor"]["mean.bias"] = payload["actor"]["mean.bias"].double()
    elif field == "config":
        payload["config"]["noise_scale"] += .1
    elif field == "optimizer":
        payload["actor_optimizer"]["param_groups"][0]["lr"] *= 2
    elif field == "replay":
        payload["extra"]["replay"]["arrays"]["reward"][0] += 1
    elif field == "rng":
        payload["rng"]["torch"][0] ^= 1
    else:
        payload["extra"].pop("checkpoint_migration")
    torch.save(payload, noise.output)
    # Bind the target identity to the tampered file so checking only file SHA
    # cannot accidentally satisfy the derived-noise equivalence requirement.
    residual.target["frozen_dsrl"]["checkpoint_sha256"] = _sha(noise.output)
    _json(residual.target_path, residual.target)
    with pytest.raises(ValueError):
        _migrate_residual(noise, residual)
    assert not residual.output.exists()


@pytest.mark.parametrize("field", ["source_sha", "source_identity", "target_identity"])
def test_frozen_chain_rejects_forged_noise_migration_provenance(frozen_chain, field):
    noise, residual = frozen_chain
    payload = _load(noise.output)
    provenance = payload["extra"]["checkpoint_migration"]
    if field == "source_sha":
        provenance["source_checkpoint"]["sha256"] = "0" * 64
    else:
        provenance[field]["inference"]["model_id"] = "different-model"
        provenance[field]["contract"]["identity"]["inference"]["model_id"] = "different-model"
        _contract_fingerprint(provenance[field])
        provenance[field.replace("identity", "schema_fingerprint")] = _schema_fingerprint(provenance[field])
    torch.save(payload, noise.output)
    residual.target["frozen_dsrl"]["checkpoint_sha256"] = _sha(noise.output)
    _json(residual.target_path, residual.target)
    with pytest.raises(ValueError):
        _migrate_residual(noise, residual)
    assert not residual.output.exists()


def test_frozen_chain_rejects_wrong_original_noise_even_when_payload_is_identical(frozen_chain, tmp_path):
    noise, residual = frozen_chain
    other = tmp_path / "reserialized-original-noise.pt"
    torch.save(_load(noise.checkpoint), other)
    assert _sha(other) != _sha(noise.checkpoint)
    with pytest.raises(ValueError):
        _migrate_residual(noise, residual, source_frozen_checkpoint=other)
    assert not residual.output.exists()
