"""Source selection must be explicit and independent of external clones."""
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from pressb import model_sources


def tree(tmp_path, family):
    marker = model_sources._FAMILIES[family][0]
    root = tmp_path / family
    (root / marker).parent.mkdir(parents=True)
    (root / marker).write_text("# source marker\n")
    return root


@pytest.fixture
def imports(monkeypatch):
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(model_sources, "_INSERTED", {})
    monkeypatch.delenv("PRESSB_WALLX_EXTENSION_DIR", raising=False)


@pytest.mark.parametrize("family", ["walloss", "vlajepa"])
def test_explicit_source_missing_marker_fails(tmp_path, family, imports):
    with pytest.raises(ValueError, match="Incomplete"):
        model_sources.activate_source(family, tmp_path)


def test_unknown_family_fails():
    with pytest.raises(ValueError, match="Unknown model family"):
        model_sources.builtin_source_root("other")


def test_activation_precedes_installed_source(tmp_path, monkeypatch, imports):
    root = tree(tmp_path, "walloss")
    (root / "packages").mkdir()
    monkeypatch.setattr(sys, "modules", {k:v for k,v in sys.modules.items() if not k.startswith(("wall_x", "openpi_client"))})
    assert model_sources.activate_source("walloss", root) == root
    assert sys.path[:2] == [str(root), str(root / "packages")]
    model_sources.activate_source("walloss", root)
    assert sys.path.count(str(root)) == 1


@pytest.mark.parametrize("name", ["starVLA", "starVLA.inference.piper_policy"])
def test_existing_external_import_is_rejected(tmp_path, monkeypatch, imports, name):
    root = tree(tmp_path, "vlajepa")
    module = ModuleType(name)
    module.__file__ = str(tmp_path / "external" / "module.py")
    monkeypatch.setitem(sys.modules, name, module)
    with pytest.raises(RuntimeError, match="another source tree"):
        model_sources.activate_source("vlajepa", root)


def test_builtin_loading_does_not_import_ml_libraries():
    root = Path(__file__).resolve().parents[1]
    code = ("import sys; from pressb.model_sources import activate_source; "
            "activate_source('walloss'); activate_source('vlajepa'); "
            "assert not any(x in sys.modules for x in ('torch','transformers','starVLA','wall_x'))")
    result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True,
                            env={**__import__('os').environ, "PYTHONPATH":str(root / 'src')})
    assert result.returncode == 0, result.stderr


def test_manifest_is_optional_for_explicit_legacy_tree(tmp_path):
    root = tree(tmp_path, "vlajepa")
    assert model_sources.source_manifest("vlajepa", root) == {}
    (root / 'source_manifest.json').write_text(json.dumps({'source_commit':'known'}))
    assert model_sources.source_manifest("vlajepa", root)['source_commit'] == 'known'


def test_explicit_extension_requires_compiled_module(tmp_path, imports):
    root = tree(tmp_path, "walloss")
    with pytest.raises(ValueError, match="No built"):
        model_sources.activate_source("walloss", root, extension_dir=tmp_path)


def test_extension_is_separate_from_python_source(tmp_path, monkeypatch, imports):
    root = tree(tmp_path, "walloss")
    extension = tmp_path/'compiled'; extension.mkdir()
    (extension/'wallx_csrc.test.so').write_bytes(b'not loaded by path selection')
    monkeypatch.setattr(sys, "modules", {k:v for k,v in sys.modules.items() if not k.startswith(("wall_x", "openpi_client", "wallx_csrc"))})
    assert model_sources.activate_source('walloss', root, extension_dir=extension) == root
    assert sys.path[0] == str(extension)


def test_source_hash_tracks_code_but_not_location_or_tests(tmp_path):
    import shutil
    root=tree(tmp_path, 'walloss')
    original=model_sources.source_tree_sha256('walloss',root)
    (root/'tests').mkdir(); (root/'tests/test_smoke.py').write_text('# tests\n')
    (root/'README.md').write_text('documentation')
    other=tmp_path/'copy'; shutil.copytree(root,other)
    assert model_sources.source_tree_sha256('walloss',other)==original
    (other/'wall_x/new.py').write_text('updated=True\n')
    assert model_sources.source_tree_sha256('walloss',other)!=original


def test_legacy_protocol_only_deployment(tmp_path, monkeypatch, imports):
    marker=tmp_path/'packages/openpi_client/websocket_policy_server.py'
    marker.parent.mkdir(parents=True); marker.write_text('# transport\n')
    monkeypatch.setattr(sys, 'modules', {k:v for k,v in sys.modules.items() if not k.startswith('openpi_client')})
    assert model_sources.activate_source('walloss',tmp_path,profile='protocol')==tmp_path
    with pytest.raises(ValueError,match='Incomplete'):
        model_sources.activate_source('walloss',tmp_path)


@pytest.mark.parametrize('different_torch', [False, True])
def test_cached_extension_matches_environment_and_source(tmp_path, monkeypatch, imports, different_torch):
    import hashlib
    import sysconfig
    root=tree(tmp_path,'walloss')
    (root/'csrc').mkdir(); source=root/'csrc/kernel.cu'; source.write_text('// kernel\n')
    extension=tmp_path/'extension'; extension.mkdir()
    binary=extension/'wallx_csrc.so'; binary.write_bytes(b'not loaded')
    receipt=extension/'receipt.json'
    receipt.write_text(json.dumps({'source_sha256':{'csrc/kernel.cu':hashlib.sha256(source.read_bytes()).hexdigest()}}))
    pointer=tmp_path/'.cache/pressb/walloss/extensions/current.json'; pointer.parent.mkdir(parents=True)
    pointer.write_text(json.dumps(dict(extension_dir=str(extension),extension_path=str(binary),
        extension_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        python_cache_tag=sys.implementation.cache_tag,python_soabi=sysconfig.get_config_var('SOABI'),
        torch_version='torch-for-test',build_receipt=str(receipt))))
    monkeypatch.setattr(model_sources,'builtin_source_root',lambda family:root)
    monkeypatch.setattr(Path,'home',classmethod(lambda cls:tmp_path))
    monkeypatch.setattr(model_sources.importlib.metadata,'version',lambda name:'different' if different_torch else 'torch-for-test')
    monkeypatch.setattr(sys, 'modules', {k:v for k,v in sys.modules.items() if not k.startswith(('wall_x','openpi_client','wallx_csrc'))})
    model_sources.activate_source('walloss')
    assert (str(extension) in sys.path) is (not different_torch)
    if not different_torch:
        source.write_text('// changed\n')
        with pytest.raises(ValueError,match='different CUDA sources'):
            model_sources.activate_source('walloss')
