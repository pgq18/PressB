"""Ascend deployment helpers must not alter system environments or DNS."""
import importlib.util
import os
from pathlib import Path
import socket
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_launcher_changes_only_its_own_local_host_resolution(monkeypatch):
    original = socket.getfqdn
    path = ROOT / 'scripts/ascend/torchrun_local.py'
    spec = importlib.util.spec_from_file_location('pressb_test_torchrun_local', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert socket.getfqdn is original  # Importing the helper has no global effect.
    queries = []
    monkeypatch.setattr(module, '_original_getfqdn', lambda host: queries.append(host) or 'remote.example')
    for name in ('', module._hostname, '0.0.0.0'):
        assert module._local_getfqdn(name) == 'localhost'
    assert module._local_getfqdn('other-host.example') == 'remote.example'
    assert queries == ['other-host.example']


def test_installer_rejects_missing_conda_environment():
    result = subprocess.run(['bash', str(ROOT / 'scripts/ascend/install_pi05.sh')],
                            env={**os.environ, 'CONDA_PREFIX': ''}, capture_output=True, text=True)
    assert result.returncode == 2
    assert 'dedicated Python 3.11 conda environment' in result.stderr


def test_installer_rejects_base_before_calling_python(tmp_path):
    prefix = tmp_path / 'conda base'
    (prefix / 'conda-meta').mkdir(parents=True)
    (prefix / 'conda-meta/history').touch()
    commands = tmp_path / 'bin'
    commands.mkdir()
    conda = commands / 'conda'
    conda.write_text('#!/bin/sh\nprintf "%s\\n" "$FAKE_CONDA_BASE"\n')
    conda.chmod(0o755)
    result = subprocess.run(['bash', str(ROOT / 'scripts/ascend/install_pi05.sh')],
                            env={**os.environ, 'CONDA_PREFIX': str(prefix),
                                 'FAKE_CONDA_BASE': str(prefix),
                                 'PATH': str(commands) + os.pathsep + os.environ['PATH']},
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert 'Refusing to install into conda base' in result.stderr
    assert not (prefix / 'bin/python').exists()
