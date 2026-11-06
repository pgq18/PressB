"""Exercise the PAP smoke runner with a CPU-only backend double."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_pap_task_backend.py"
spec = importlib.util.spec_from_file_location("pap_task_smoke", SCRIPT)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


@pytest.fixture
def config_path(tmp_path):
    scene = tmp_path / "scene.json"
    scene.write_text(json.dumps({"fps": 30, "physics_dt": 1 / 120}))
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"backend": {"factory": "unused:factory", "config": {
        "scene_config": str(scene), "workers": [{"gpu": 0}], "num_envs": 1,
        "max_episode_seconds": 20, "render_resolution": [640, 480],
        "render_subframes": 1, "reset_render_frames": 2,
        "smoothing": {"mode": "none", "alpha": .8},
        "scene_overrides": {"physics_dt": 1 / 120},
    }}}))
    return path


def arguments(config_path, output):
    return smoke.parser().parse_args(["--config", str(config_path), "--output-dir", str(output)])


def test_help_does_not_import_pressb_or_simulator():
    script = """
import importlib.abc
import runpy
import sys

class RejectBackendImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("pressb", "isaacsim", "omni", "carb", "pxr"):
            raise AssertionError("Unexpected backend import: " + fullname)

sys.meta_path.insert(0, RejectBackendImports())
sys.argv = [sys.argv[1], "--help"]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
    result = subprocess.run([sys.executable, "-c", script, str(SCRIPT)],
        capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    assert "--num-envs" in result.stdout and "--gpu-id" in result.stdout


def test_defaults_and_resolved_config_preserve_physics_and_source_file(config_path, tmp_path):
    args = arguments(config_path, tmp_path / "run")
    original = json.loads(config_path.read_text())

    config = smoke.resolved_config(args, args.output_dir)

    assert args.num_envs == config["num_envs"] == 2
    assert args.gpu_id == config["gpu"] == 1
    assert config["gpu_ids"] == [1]
    assert "workers" not in config
    assert config["max_episode_seconds"] == pytest.approx(2 / 30)
    assert config["python"] == sys.executable
    assert config["log_dir"] == str(args.output_dir / "workers")
    for key in ("scene_overrides", "smoothing", "render_resolution", "render_subframes", "reset_render_frames"):
        assert config[key] == original["backend"]["config"][key]
    assert json.loads(config_path.read_text()) == original


@pytest.mark.parametrize("option,value", [("--num-envs", "0"), ("--gpu-id", "-1"), ("--seed", "-1")])
def test_invalid_command_line_counts_fail_before_loading_backend(option, value):
    with pytest.raises(SystemExit) as error:
        smoke.parser().parse_args([option, value])
    assert error.value.code == 2


class FakeBackend:
    def __init__(self, config, *, fail_step=False, omit_reset=False):
        self.config = config
        self.closed = False
        self.calls = []
        self.timings = {"world_ticks": 0, "captures": 0}
        self.fail_step = fail_step
        self.omit_reset = omit_reset

    def describe(self):
        self.calls.append("describe")
        return {"task_id": "pap", "num_envs": self.config["num_envs"],
            "capabilities": ["step_and_reset"],
            "action": {"lower": [-2.] * 6 + [0.], "upper": [2.] * 6 + [.08]},
            "source_identity": {"tasks/pap/vector.py": {"sha256": "cpu-fixture", "bytes": 1}},
            "source_revision": "sha256:cpu-fixture", "execution": {"backend": "cpu_test_double"}}

    def rows(self, *, images=True, tick=0):
        image = {"encoding": "png_base64", "data": "cpu-fixture"}
        return [{"env_id": i, "observation": {
            "state": [i * .1 + tick * .01] * 6 + [.04],
            "images": {"up": image.copy(), "wrist": image.copy()} if images else {},
        }, "info": {"physics_steps": 4 if tick else 0, "episode_physics_steps": tick}}
            for i in range(self.config["num_envs"])]

    def reset(self, ids, seed, options):
        self.calls.append(("reset", ids, seed, options))
        self.timings["captures"] += 2
        return self.rows()

    def step(self, items):
        self.calls.append(("step", deepcopy(items)))
        if self.fail_step:
            raise RuntimeError("synthetic step failure")
        self.timings["world_ticks"] += 4
        self.timings["captures"] += 2
        rows = self.rows(tick=4)
        for row in rows:
            row.update(terminated=False, truncated=False)
        return rows

    def step_and_reset(self, items, *, reset_seed, reset_options, max_active_envs):
        self.calls.append(("fused", deepcopy(items), reset_seed, reset_options, max_active_envs))
        self.timings["world_ticks"] += 4
        self.timings["captures"] += 2
        terminal = self.rows(images=False, tick=8)
        for row in terminal:
            row.update(terminated=False, truncated=True, terminal_observation_omitted=True)
            row["info"]["termination_reason"] = "timeout"
        return {"items": terminal, "reset_items": [] if self.omit_reset else self.rows()}

    def close(self):
        self.calls.append("close")
        self.closed = True


def test_runner_uses_latest_measured_hold_and_checks_fused_timeout_reset(config_path, tmp_path, capsys):
    args = arguments(config_path, tmp_path / "run")
    instances = []

    def factory(config):
        instances.append(FakeBackend(config))
        return instances[-1]

    assert smoke.run(args, backend_factory=factory) == 0

    backend = instances[0]
    assert backend.closed
    assert backend.calls[0] == "describe" and backend.calls[-1] == "close"
    step, fused = backend.calls[2], backend.calls[3]
    assert step[1][0]["actions"] == [[0.] * 6 + [.04]]
    assert fused[1][0]["actions"] == [[.04] * 6 + [.04]]
    assert fused[2:] == (43, {}, 2)
    report = json.loads((args.output_dir / "report.json").read_text())
    assert report["status"] == "passed" and report["closed"]
    assert report["fused"]["timeout_env_ids"] == [0, 1]
    assert report["fused"]["reset_env_ids"] == [0, 1]
    assert report["source_identity"] == backend.describe()["source_identity"]
    assert report["timing"]["backend"]["world_ticks"] == 8
    assert "close_seconds" in report["timing"]
    assert (args.output_dir / "metadata.json").is_file()
    assert json.loads(capsys.readouterr().out)["status"] == "passed"


@pytest.mark.parametrize("failure,error_text", [
    ({"fail_step": True}, "synthetic step failure"),
    ({"omit_reset": True}, "omitted or duplicated"),
])
def test_runner_reports_failures_and_always_closes_backend(config_path, tmp_path, failure, error_text):
    args = arguments(config_path, tmp_path / "failed")
    instances = []

    def factory(config):
        instances.append(FakeBackend(config, **failure))
        return instances[-1]

    assert smoke.run(args, backend_factory=factory) == 1

    assert instances[0].closed
    report = json.loads((args.output_dir / "report.json").read_text())
    assert report["status"] == "failed" and report["closed"]
    assert error_text in report["error"]


def test_success_evidence_is_written_before_close_can_exit_process(config_path, tmp_path):
    args = arguments(config_path, tmp_path / "fast_shutdown")

    class ExitingBackend(FakeBackend):
        def close(self):
            recorded = json.loads((args.output_dir / "report.json").read_text())
            assert recorded["operations_status"] == "passed"
            assert recorded["status"] == "closing"
            assert recorded["close_attempted"] and not recorded["closed"]
            assert recorded["fused"]["reset_env_ids"] == [0, 1]
            raise SystemExit(0)

    with pytest.raises(SystemExit) as error:
        smoke.run(args, backend_factory=ExitingBackend)
    assert error.value.code == 0
    report = json.loads((args.output_dir / "report.json").read_text())
    assert report["status"] == "closing" and not report["closed"]


def test_output_is_unique_and_existing_directory_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "ROOT", tmp_path)
    first = smoke._output_directory(None)
    second = smoke._output_directory(None)
    assert first != second
    assert first.parent == second.parent == tmp_path / "outputs/pap_task_smoke"
    marker = first / "keep.txt"
    marker.write_text("untouched")
    with pytest.raises(FileExistsError):
        smoke._output_directory(first)
    assert marker.read_text() == "untouched"
