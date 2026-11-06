"""Model source selection must not silently depend on another checkout."""
from __future__ import annotations

import hashlib
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("torch")

from pressb.models import vlajepa, vlajepa_joint, walloss
from pressb.models import vlajepa_backend as inference
from pressb.compat.elevator_rl import inference_transport


class SourceSelected(Exception):
    pass


@pytest.mark.parametrize("external", [None, "/explicit/model/source"])
@pytest.mark.parametrize("adapter", ["walloss", "pose", "joint", "websocket"])
def test_adapter_selects_bundled_default_or_explicit_source(monkeypatch, external, adapter):
    calls = []

    def activate(family, root=None, **kwargs):
        calls.append((family, root))
        raise SourceSelected

    if adapter == "walloss":
        monkeypatch.setattr(walloss, "activate_source", activate)
        call = lambda: walloss.WallOSSBackend({} if external is None else {"repo": external})
        family = "walloss"
    elif adapter == "pose":
        monkeypatch.setattr(inference, "activate_source", activate)
        call = lambda: inference.FrozenPiperBackend.load(external, "/checkpoint")
        family = "vlajepa"
    elif adapter == "joint":
        monkeypatch.setattr(vlajepa_joint, "activate_source", activate)
        monkeypatch.setattr(vlajepa_joint, "read_checkpoint_metadata", lambda *args, **kwargs: {})
        config = {"run": "/run"}
        if external is not None:
            config["vla_repo"] = external
        call = lambda: vlajepa_joint.create_backend(config)
        family = "vlajepa"
    else:
        monkeypatch.setattr(inference_transport, "activate_source", activate)
        call = lambda: inference_transport.load_wallx_server(external)
        family = "walloss"
    with pytest.raises(SourceSelected):
        call()
    assert calls == [(family, external)]


def test_pose_factory_does_not_require_external_repo(monkeypatch):
    calls = []
    frozen = object()
    monkeypatch.setattr(inference.FrozenPiperBackend, "load", lambda **kwargs: calls.append(kwargs) or frozen)
    monkeypatch.setattr(vlajepa, "VLAJEPABackend", lambda backend: backend)
    assert vlajepa.create_backend({"checkpoint": "/checkpoint"}) is frozen
    assert calls == [{"checkpoint": "/checkpoint"}]
    with pytest.raises(ValueError, match="requires checkpoint"):
        vlajepa.create_backend({})


def test_walloss_forwards_explicit_extension_build_directory(monkeypatch):
    calls = []

    def activate(family, root=None, *, extension_dir=None):
        calls.append((family, root, extension_dir))
        raise SourceSelected

    monkeypatch.setattr(walloss, "activate_source", activate)
    with pytest.raises(SourceSelected):
        walloss.WallOSSBackend({"extension_dir": "/separate/cuda/build"})
    assert calls == [("walloss", None, "/separate/cuda/build")]


def test_legacy_protocol_only_deployment_remains_loadable(monkeypatch, tmp_path):
    source = tmp_path / "packages/openpi_client/websocket_policy_server.py"
    source.parent.mkdir(parents=True)
    source.write_text("# Historical deployment copied protocol files only.\n")
    server = object()
    module = SimpleNamespace(__file__=str(source), WebsocketPolicyServer=server)
    for name in tuple(sys.modules):
        if name == "openpi_client" or name.startswith("openpi_client."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(inference_transport.importlib, "import_module", lambda name: module)
    assert inference_transport.load_wallx_server(tmp_path) is server


def test_frozen_loader_requires_checkpoint_before_source_selection(monkeypatch):
    monkeypatch.setattr(inference, "activate_source", lambda *args: pytest.fail("Selected source without checkpoint"))
    with pytest.raises(ValueError, match="checkpoint is required"):
        inference.FrozenPiperBackend.load()


@pytest.mark.parametrize("bundled", [True, False])
def test_source_identity_distinguishes_bundled_manifest_from_external_git(monkeypatch, tmp_path, bundled):
    root = tmp_path / "model"
    files = ("starVLA/inference/piper_policy.py", "starVLA/model/framework/VLA_JEPA.py",
             "starVLA/model/modules/action_model/GR00T_ActionHeader.py", "starVLA/model/modules/vlm/QWen3.py")
    for name in files:
        source = root / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(name)
    manifest = root / "source_manifest.json"
    manifest.write_text('{"upstream_commit":"archived-vla-revision"}')
    monkeypatch.setattr(inference, "builtin_source_root", lambda family: root if bundled else tmp_path / "builtin")
    monkeypatch.setattr(inference, "source_tree_sha256", lambda family, root: "live-model-source-digest")
    calls = []

    def git(*args, **kwargs):
        calls.append(args)
        return "external-commit\n"

    monkeypatch.setattr(inference.subprocess, "check_output", git)
    result = inference._source_identity(root)
    assert set(result["source_sha256"]) == set(files)
    assert result["vla_repo"] == str(root)
    assert result["addon_sha256"] == hashlib.sha256(Path(inference.__file__).read_bytes()).hexdigest()
    if bundled:
        assert not calls
        assert result["git_commit"] is None
        assert result["source_distribution"] == "pressb_builtin"
        assert result["vendored_manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
        assert result["bundled_model_source_sha256"] == "live-model-source-digest"
    else:
        assert len(calls) == 1
        assert result["git_commit"] == "external-commit"
        assert set(result) == {"vla_repo", "git_commit", "source_sha256", "addon_sha256"}


@pytest.mark.parametrize("family,module_name", [("walloss", "wall_x"), ("vlajepa", "starVLA")])
def test_bundled_selection_rejects_already_imported_external_model(monkeypatch, tmp_path, family, module_name):
    from pressb.model_sources import activate_source

    foreign = ModuleType(module_name)
    foreign.__file__ = str(tmp_path / module_name / "__init__.py")
    foreign.__path__ = [str(tmp_path / module_name)]
    monkeypatch.setitem(sys.modules, module_name, foreign)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    with pytest.raises((RuntimeError, ImportError), match="[Aa]lready|[Dd]ifferent|[Oo]utside|[Cc]onflict|[Ff]oreign"):
        activate_source(family)


def test_inference_deployment_contains_builtin_sources_without_external_checkouts(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    module = runpy.run_path(str(root / "scripts/sync_rl_inference.py"))
    calls = []
    monkeypatch.setattr(module["subprocess"], "run", lambda command, **kwargs: calls.append((command, kwargs)))
    monkeypatch.setattr(sys, "argv", ["sync_rl_inference.py", "--host", "test-host", "--remote-root", "/deploy/PressB",
                                      "--ssh-socket", "", "--dry-run"])
    module["main"]()
    assert len(calls) == 1
    command, options = calls[0]
    assert command[0] == "rsync" and "--dry-run" in command
    assert "src/pressb/" in command
    assert "scripts/serve_rl_inference.py" in command
    assert "scripts/serve_inference_node.py" in command
    assert "--exclude=__pycache__/" in command
    assert not any(path.startswith(("ref/", "/home/pengguanqi/Models")) for path in command)
    assert all((root / name).exists() for name in module["FILES"])
    assert options["cwd"] == root
