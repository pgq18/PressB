"""Explicit, offline source-revision migration of trusted SAC checkpoints.

This checks schema equality and exact saved training state, not equivalence of
arbitrary source code. The caller must approve source revision changes and
supply a regression report. Normal trainer/checkpoint loading stays strict.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

import numpy as np
import torch

from .contract import fingerprint
from .learner import CHECKPOINT_VERSION, SOURCE_COMMIT, LearnerConfig
from .trainer import _canonical, _schema_fingerprint


LIMITATION = ("Schema equality and unchanged checkpoint values do not prove that arbitrary "
              "source revisions have equivalent semantics. The caller explicitly approved "
              "these revisions using the referenced regression evidence.")
_RUNTIME_FIELDS = {"num_envs", "fingerprint", "terminal_observation_mode", "fused_step_reset"}
_REVISION_PATHS = {("inference", "source_revision"), ("simulation", "source_revision"),
                   ("contract", "identity", "inference", "source_revision"),
                   ("contract", "identity", "simulation", "source_revision")}
_FROZEN_PATHS = {("frozen_dsrl", "checkpoint_sha256"), ("frozen_dsrl", "schema_fingerprint")}
_MISSING = object()


def _stat_key(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _file_record(path):
    path = Path(path).expanduser().resolve(strict=True)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if _stat_key(before) != _stat_key(path.stat()):
        raise ValueError(f"Input changed while reading: {path}")
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": before.st_size}


def _json_file(path):
    record = _file_record(path)

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    data = Path(record["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError("Identity file changed while reading")
    value = json.loads(data, object_pairs_hook=unique_keys,
                       parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON: {token}")))
    if not isinstance(value, dict):
        raise ValueError("Negotiated identity must be a JSON object")
    return value, record


def _checkpoint(path):
    record = _file_record(path)
    before = Path(record["path"]).stat()
    # Checkpoints are trusted pickle inputs. Map tensors to CPU and do not
    # instantiate a learner, which would restore/change process RNG state.
    payload = torch.load(record["path"], map_location="cpu", weights_only=False, mmap=True)
    if _stat_key(before) != _stat_key(Path(record["path"]).stat()):
        raise ValueError("Checkpoint changed while loading")
    required = {"version", "source_commit", "config", "updates", "actor", "qs", "q_targets",
                "log_alpha", "actor_optimizer", "q_optimizer", "alpha_optimizer", "rng", "extra"}
    if (not isinstance(payload, dict) or not required.issubset(payload)
            or payload["version"] != CHECKPOINT_VERSION or payload["source_commit"] != SOURCE_COMMIT
            or not isinstance(payload["extra"], dict)):
        raise ValueError("Unsupported or incomplete learner checkpoint")
    LearnerConfig(**payload["config"])
    return payload, record


def _assert_same(first, second, path="payload"):
    """Check exact tensor/array bytes and recursively compare all other state."""
    if type(first) is not type(second):
        raise ValueError(f"Saved state changed at {path}: type differs")
    if isinstance(first, torch.Tensor):
        if (first.dtype != second.dtype or first.shape != second.shape
                or first.layout != second.layout or first.stride() != second.stride()
                or first.requires_grad != second.requires_grad):
            raise ValueError(f"Saved tensor state changed at {path}")
        a = first.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        b = second.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        if any(not torch.equal(a[i:i + 8_388_608], b[i:i + 8_388_608])
               for i in range(0, a.numel(), 8_388_608)):
            raise ValueError(f"Saved tensor values changed at {path}")
    elif isinstance(first, np.ndarray):
        if first.dtype != second.dtype or first.shape != second.shape or first.dtype.hasobject:
            raise ValueError(f"Saved array state changed at {path}")
        a = np.ascontiguousarray(first).view(np.uint8).reshape(-1)
        b = np.ascontiguousarray(second).view(np.uint8).reshape(-1)
        if any(not np.array_equal(a[i:i + 8_388_608], b[i:i + 8_388_608])
               for i in range(0, a.size, 8_388_608)):
            raise ValueError(f"Saved array values changed at {path}")
    elif isinstance(first, Mapping):
        if list(first) != list(second):
            raise ValueError(f"Saved mapping keys changed at {path}")
        for key in first:
            _assert_same(first[key], second[key], f"{path}.{key}")
        if hasattr(first, "__dict__"):
            _assert_same(vars(first), vars(second), f"{path}.__dict__")
    elif isinstance(first, (list, tuple)):
        if len(first) != len(second):
            raise ValueError(f"Saved sequence length changed at {path}")
        for index, (a, b) in enumerate(zip(first, second, strict=True)):
            _assert_same(a, b, f"{path}[{index}]")
    elif isinstance(first, float):
        if struct.pack("!d", first) != struct.pack("!d", second):
            raise ValueError(f"Saved scalar changed at {path}")
    elif isinstance(first, np.generic):
        if first.dtype != second.dtype or first.tobytes() != second.tobytes():
            raise ValueError(f"Saved scalar changed at {path}")
    elif first != second:
        raise ValueError(f"Saved value changed at {path}")


def _differences(old, new, path=()):
    if isinstance(old, dict) and isinstance(new, dict):
        result = []
        for key in sorted(old.keys() | new.keys()):
            result.extend(_differences(old.get(key, _MISSING), new.get(key, _MISSING), (*path, key)))
        return result
    if old is not _MISSING and new is not _MISSING and _canonical(old) == _canonical(new):
        return []
    return [{"path": list(path), "old": None if old is _MISSING else old,
             "new": None if new is _MISSING else new,
             "old_present": old is not _MISSING, "new_present": new is not _MISSING}]


def _validate_identity(identity):
    for name in ("contract", "inference", "simulation", "action_projection"):
        if not isinstance(identity.get(name), dict):
            raise ValueError(f"Negotiated identity requires {name}")
    contract = identity["contract"]
    expected = fingerprint({key: value for key, value in contract.items()
                            if key not in ("num_envs", "fingerprint")})
    if contract.get("fingerprint") != expected:
        raise ValueError("Negotiated contract fingerprint does not match its contents")
    mirrors = contract.get("identity", {})
    for role in ("inference", "simulation"):
        if not isinstance(mirrors.get(role), dict):
            raise ValueError(f"Negotiated contract.identity requires {role}")
        outer = identity[role].get("source_revision", _MISSING)
        inner = mirrors[role].get("source_revision", _MISSING)
        if outer != inner or (outer is not _MISSING and (not isinstance(outer, str) or not outer)):
            raise ValueError(f"Inconsistent {role}.source_revision mirror")


def _validate_change(old, new, *, frozen=False):
    _validate_identity(old)
    _validate_identity(new)
    differences = _differences(old, new)
    revision_changes = 0
    for difference in differences:
        path = tuple(difference["path"])
        if path in _REVISION_PATHS:
            if not difference["old_present"] or not difference["new_present"]:
                raise ValueError("Source revision fields cannot be added or removed")
            revision_changes += 1
        elif len(path) == 2 and path[0] == "contract" and path[1] in _RUNTIME_FIELDS:
            # Exactly the same normalization as the strict trainer fingerprint.
            # Extra terminal/fused replay flags remain untouched by migration.
            continue
        elif frozen and path in _FROZEN_PATHS:
            continue
        else:
            raise ValueError(f"Semantic identity change is forbidden: {'.'.join(path)}")
    if not revision_changes:
        raise ValueError("No source revision changes to migrate")
    return differences


def _validate_source(payload, identity):
    if payload["extra"].get("schema_fingerprint") != _schema_fingerprint(identity):
        raise ValueError("Source checkpoint schema_fingerprint does not match source identity")
    if _canonical(payload["extra"].get("contract")) != _canonical(identity["contract"]):
        raise ValueError("Source checkpoint contract does not match source identity")
    expected_method = "dsrl" if payload["config"]["method"] == "initial_noise" else "action_residual"
    if identity["contract"].get("method") != expected_method:
        raise ValueError("Checkpoint method does not match negotiated contract")


def _derived_payload(source, target_identity, migration, *, frozen_provenance=None):
    # Preserve tensors, optimizers, replay arrays, RNG objects, and all unknown
    # extra fields without normalization, casting, or rebuilding a learner.
    result = dict(source)
    result["extra"] = dict(source["extra"])
    result["extra"].update(schema_fingerprint=_schema_fingerprint(target_identity),
                           contract=copy.deepcopy(target_identity["contract"]),
                           checkpoint_migration=migration)
    if frozen_provenance is not None:
        result["extra"]["frozen_dsrl"] = frozen_provenance
    return result


def _actor_sha256(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(_canonical([name, str(array.dtype), list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _noise_identity(residual):
    identity = copy.deepcopy(residual)
    identity.pop("frozen_dsrl", None)
    identity["contract"]["method"] = "dsrl"
    identity["contract"]["fingerprint"] = fingerprint({
        key: value for key, value in identity["contract"].items() if key not in ("num_envs", "fingerprint")})
    return identity


def _validate_frozen_chain(old, new, source_path, target_path):
    source, source_file = _checkpoint(source_path)
    target, target_file = _checkpoint(target_path)
    if any(item["config"]["method"] != "initial_noise" or "frozen_dsrl" in item["extra"]
           for item in (source, target)):
        raise ValueError("Frozen chain requires initial_noise checkpoints")
    migration = target["extra"].get("checkpoint_migration")
    if (not isinstance(migration, dict) or migration.get("format_version") != 1
            or migration.get("kind") != "source_revision_only"
            or migration.get("allow_source_revision_change") is not True
            or migration.get("source_checkpoint", {}).get("sha256") != source_file["sha256"]
            or migration.get("source_checkpoint", {}).get("bytes") != source_file["bytes"]):
        raise ValueError("Derived frozen checkpoint has no matching migration source provenance")
    source_identity, target_identity = migration.get("source_identity"), migration.get("target_identity")
    if not isinstance(source_identity, dict) or not isinstance(target_identity, dict):
        raise ValueError("Frozen checkpoint migration identities are missing")
    differences = _validate_change(source_identity, target_identity)
    _validate_source(source, source_identity)
    _validate_source(target, target_identity)
    if (migration.get("differences") != differences
            or migration.get("source_schema_fingerprint") != _schema_fingerprint(source_identity)
            or migration.get("target_schema_fingerprint") != _schema_fingerprint(target_identity)):
        raise ValueError("Frozen migration identity provenance is inconsistent")
    if migration.get("previous_migration") != source["extra"].get("checkpoint_migration"):
        raise ValueError("Frozen migration discarded its previous provenance")
    _assert_same(_derived_payload(source, target_identity, migration), target, "frozen checkpoint")
    actor_sha = _actor_sha256(source["actor"])
    for identity, noise_identity, noise_payload, file_record in (
            (old, source_identity, source, source_file), (new, target_identity, target, target_file)):
        frozen = identity.get("frozen_dsrl", {})
        if (frozen.get("checkpoint_sha256") != file_record["sha256"]
                or frozen.get("schema_fingerprint") != _schema_fingerprint(noise_identity)
                or _schema_fingerprint(_noise_identity(identity)) != _schema_fingerprint(noise_identity)
                or _canonical(frozen.get("learner_config")) != _canonical(asdict(LearnerConfig(**noise_payload["config"])))):
            raise ValueError("Frozen checkpoint reference/config/schema does not match the residual identity")
    return {"source_checkpoint": source_file, "target_checkpoint": target_file,
            "actor_sha256": actor_sha, "source_updates": source["updates"],
            "source_transitions": source["extra"].get("transitions"),
            "source_schema_fingerprint": _schema_fingerprint(source_identity),
            "target_schema_fingerprint": _schema_fingerprint(target_identity)}


def _updated_frozen_provenance(payload, old, chain):
    provenance = payload["extra"].get("frozen_dsrl")
    if not isinstance(provenance, dict):
        raise ValueError("Residual checkpoint has no frozen DSRL provenance")
    if any(_canonical(provenance.get(key)) != _canonical(value)
           for key, value in old["frozen_dsrl"].items()):
        raise ValueError("Residual frozen provenance does not match source identity")
    if (provenance.get("actor_sha256") != chain["actor_sha256"]
            or provenance.get("updates") != chain["source_updates"]
            or provenance.get("transitions") != chain["source_transitions"]):
        raise ValueError("Residual frozen actor/update provenance does not match the source checkpoint")
    verification = provenance.get("verification", {})
    if verification and (verification.get("actor_sha256") != chain["actor_sha256"]
                         or verification.get("matches_initial") is not True):
        raise ValueError("Residual frozen actor verification is inconsistent")
    updated = copy.deepcopy(provenance)
    updated.update(checkpoint=chain["target_checkpoint"]["path"],
                   checkpoint_sha256=chain["target_checkpoint"]["sha256"],
                   schema_fingerprint=chain["target_schema_fingerprint"])
    return updated


def migrate_checkpoint(source_checkpoint, source_identity, target_identity, output, *,
                       allow_source_revision_change=False, regression_report,
                       source_frozen_checkpoint=None, target_frozen_checkpoint=None):
    """Write a new derived checkpoint and ``<output>.migration.json`` report.

    All inputs are file paths. Inputs must be trusted local checkpoints because
    PyTorch pickle loading can execute code. This function never connects to a
    service or GPU, and never changes normal loader compatibility checks.
    """
    if allow_source_revision_change is not True:
        raise ValueError("Explicit allow_source_revision_change=True approval is required")
    source_file = Path(source_checkpoint).expanduser().resolve(strict=True)
    output = Path(output).expanduser().absolute()
    if output.resolve() == source_file:
        raise ValueError("Output must not overwrite the source checkpoint")
    output = output.parent.resolve() / output.name
    report_path = Path(str(output) + ".migration.json")
    for path in (output, report_path):
        if os.path.lexists(path):
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    evidence = _file_record(regression_report)
    if not evidence["bytes"]:
        raise ValueError("Regression report must not be empty")
    old, old_file = _json_file(source_identity)
    new, new_file = _json_file(target_identity)
    has_frozen = "frozen_dsrl" in old or "frozen_dsrl" in new
    differences = _validate_change(old, new, frozen=has_frozen)
    chain = None
    if has_frozen:
        if ("frozen_dsrl" not in old or "frozen_dsrl" not in new
                or source_frozen_checkpoint is None or target_frozen_checkpoint is None):
            raise ValueError("Residual migration requires both source and derived frozen checkpoints")
        # Release both full noise payloads before loading the residual replay.
        # mmap maps torch storage, but NumPy replay still occupies CPU memory.
        chain = _validate_frozen_chain(old, new, source_frozen_checkpoint, target_frozen_checkpoint)
    elif source_frozen_checkpoint is not None or target_frozen_checkpoint is not None:
        raise ValueError("Frozen checkpoints/provenance require a residual frozen identity")
    payload, checkpoint_file = _checkpoint(source_file)
    _validate_source(payload, old)
    frozen_provenance = _updated_frozen_provenance(payload, old, chain) if chain else None
    if not has_frozen and "frozen_dsrl" in payload["extra"]:
        raise ValueError("Frozen checkpoints/provenance require a residual frozen identity")
    migration = {"format_version": 1, "kind": "source_revision_only",
                 "created_at_utc": datetime.now(timezone.utc).isoformat(),
                 "source_checkpoint": checkpoint_file, "source_identity_file": old_file,
                 "target_identity_file": new_file, "regression_report": evidence,
                 "source_identity": old, "target_identity": new, "differences": differences,
                 "source_schema_fingerprint": _schema_fingerprint(old),
                 "target_schema_fingerprint": _schema_fingerprint(new),
                 "allow_source_revision_change": True, "limitation": LIMITATION}
    if "checkpoint_migration" in payload["extra"]:
        migration["previous_migration"] = payload["extra"]["checkpoint_migration"]
    if chain is not None:
        migration["frozen_chain"] = chain
    derived = _derived_payload(payload, new, migration, frozen_provenance=frozen_provenance)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_report = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(derived, stream)
            stream.flush()
            os.fsync(stream.fileno())
        saved, output_record = _checkpoint(temporary)
        _assert_same(derived, saved)
        output_record["path"] = str(output)
        report = {**migration, "output_checkpoint": output_record, "report_path": str(report_path)}
        with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, prefix=f".{report_path.name}.", delete=False) as stream:
            temporary_report = Path(stream.name)
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        inputs = [checkpoint_file, old_file, new_file, evidence]
        if chain:
            inputs.extend((chain["source_checkpoint"], chain["target_checkpoint"]))
        for record in inputs:
            if _file_record(record["path"]) != record:
                raise ValueError(f"Input changed during migration: {record['path']}")
        # Hard-link publication is atomic and fails if a competing writer has
        # created either path; unlike replace(), it cannot overwrite anything.
        os.link(temporary, output)
        os.link(temporary_report, report_path)
        return report
    except BaseException:
        # An interrupt can arrive after either link succeeded, including before
        # Python could update a separate publication flag. Remove only links to
        # our own temporary inodes; never remove another writer's destination.
        for source, destination in ((temporary, output), (temporary_report, report_path)):
            if source is not None and destination.exists() and os.path.samefile(source, destination):
                destination.unlink()
        raise
    finally:
        for path in (temporary, temporary_report):
            if path is not None and path.exists():
                path.unlink()
