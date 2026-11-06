"""Video audits bind each source hash to its selected current or frozen tree."""
import hashlib
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "_video_source_provenance", ROOT / "scripts/audit_two_button_video.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_source(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path, hashlib.sha256(content).hexdigest()


def source_tree(root, *, historical=False):
    owner = "src/pressb/online_rl" if historical else "src/pressb/compat/elevator_rl"
    paths, hashes = {}, {}
    names = {
        "runner": owner + "/fast_runner.py",
        "learner": "src/pressb/online_rl/learner.py" if historical else "src/pressb/runtime/learner.py",
        "simulation": "src/pressb/online_rl/simulation.py" if historical else "src/pressb/tasks/elevator/simulation.py",
        "adapter": "src/pressb/online_rl/inference.py" if historical else "src/pressb/models/vlajepa_backend.py",
        "vla": "model_source/starVLA/inference/piper_policy.py",
    }
    for role, name in names.items():
        content = ("# " + ("historical " if historical else "current ") + role + "\n").encode()
        paths[role], hashes[role] = write_source(root, name, content)
    manifest = {
        "sources": {
            "fast_runner.py": hashes["runner"],
            "learner.py" if historical else names["learner"]: hashes["learner"],
        },
        "identities": {
            "simulation": {"source_sha256": {names["simulation"]: hashes["simulation"]}},
            "inference": {"source_identity": {
                "addon_sha256": hashes["adapter"],
                "vla_repo": "model_source",
                "source_sha256": {"starVLA/inference/piper_policy.py": hashes["vla"]},
            }},
        },
    }
    return manifest, paths


def test_current_tree_resolves_runner_basenames_and_canonical_qualified_paths(audit, tmp_path):
    manifest, paths = source_tree(tmp_path)
    assert audit.source_path("fast_runner.py", tmp_path, runner=True) == paths["runner"]
    assert audit.source_path("src/pressb/runtime/learner.py", tmp_path, runner=True) == paths["learner"]
    assert audit.verify_sources(manifest, tmp_path) is None


def test_explicit_historical_tree_verifies_original_bytes_despite_changed_current_sources(audit, tmp_path, monkeypatch):
    current, archive = tmp_path / "checkout", tmp_path / "run/source"
    _, current_paths = source_tree(current)
    manifest, archived_paths = source_tree(archive, historical=True)
    monkeypatch.setattr(audit, "ROOT", current)
    assert current_paths["runner"].read_bytes() != archived_paths["runner"].read_bytes()
    assert audit.source_path("fast_runner.py", archive, runner=True) == archived_paths["runner"]
    assert audit.verify_sources(manifest, archive) is None
    with pytest.raises(ValueError, match="Runner source changed"):
        audit.verify_sources(manifest, current)


@pytest.mark.parametrize("historical", [False, True], ids=["current", "historical"])
@pytest.mark.parametrize("role,message", [
    ("runner", "Runner source changed"),
    ("learner", "Runner source changed"),
    ("simulation", "Simulation source changed"),
    ("adapter", "inference adapter differs"),
    ("vla", "VLA source changed"),
])
def test_changed_source_bytes_are_rejected_in_every_manifest_section(audit, tmp_path, historical, role, message):
    manifest, paths = source_tree(tmp_path, historical=historical)
    audit.verify_sources(manifest, tmp_path)
    paths[role].write_bytes(paths[role].read_bytes() + b"# changed after recording\n")
    with pytest.raises(ValueError, match=message):
        audit.verify_sources(manifest, tmp_path)


def test_qualified_historical_path_never_substitutes_a_matching_current_file(audit, tmp_path):
    manifest, paths = source_tree(tmp_path)
    digest = manifest["identities"]["simulation"]["source_sha256"].pop(
        "src/pressb/tasks/elevator/simulation.py")
    historical_name = "src/pressb/online_rl/simulation.py"
    manifest["identities"]["simulation"]["source_sha256"][historical_name] = digest
    assert paths["simulation"].is_file()
    assert audit.source_path(historical_name, tmp_path) == tmp_path / historical_name
    with pytest.raises(FileNotFoundError):
        audit.verify_sources(manifest, tmp_path)


def test_copied_absolute_checkout_paths_rebase_but_external_model_paths_stay_explicit(audit, tmp_path, monkeypatch):
    current, archive = tmp_path / "checkout", tmp_path / "run/source"
    current.mkdir()
    archive.mkdir(parents=True)
    monkeypatch.setattr(audit, "ROOT", current)
    name = "src/pressb/tasks/elevator/simulation.py"
    archived, _ = write_source(archive, name, b"# archived simulation\n")
    external, digest = write_source(tmp_path, "external_model/policy.py", b"# external model\n")
    assert audit.source_path(current / name, archive) == archived
    assert audit.source_path(external, archive) == external
    assert audit.sha(audit.source_path(external, archive)) == digest


def test_relative_source_path_cannot_escape_the_selected_tree(audit, tmp_path):
    with pytest.raises(ValueError, match="escapes selected source tree"):
        audit.source_path("../another_checkout/policy.py", tmp_path)
