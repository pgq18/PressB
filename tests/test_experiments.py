"""Plugin discovery and experiment compilation require no simulator/model load."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from pressb.experiments.config import load_experiment, materialize, PROJECT_ROOT
from pressb.experiments.cli import main
from pressb.registry import available, resolve


def minimal():
    return dict(schema_version=1, name="test", task={"plugin": "pap", "config": {}},
                model={"plugin": "walloss", "config": {}}, run={"method": "base", "mode": "eval"})


def save(path, value):
    path.write_text(json.dumps(value))
    return path


def test_discovery_lazy_in_fresh_process():
    script = "from pressb.registry import available; import sys; assert len(available()['task']) >= 4; assert not any(x in sys.modules for x in ['torch','isaacsim']); assert not any(x.startswith(('pressb.tasks.', 'pressb.models.')) for x in sys.modules)"
    subprocess.run([sys.executable, "-c", script], check=True,
                   env={**__import__('os').environ, "PYTHONPATH": str(PROJECT_ROOT/'src')})


def test_native_parallel_defaults_do_not_replace_compatibility_plugins():
    tasks = available("task")
    for name in ("pap", "elevator", "table_buttons", "playground"):
        assert tasks[name] == f"pressb.tasks.{name}.parallel:create_backend"
    assert tasks["pap_process"] == "pressb.tasks.pap.backend:create_backend"
    assert tasks["pap_recording"] == "pressb.tasks.pap.recording:create_backend"
    assert tasks["elevator_remote"] == "pressb.tasks.elevator.backend:create_backend"
    assert tasks["piper_real"] == "pressb.runtime.real_robot:create_backend"
    tasks["pap"] = "external:factory"
    assert resolve("task", "pap") == "pressb.tasks.pap.parallel:create_backend"


@pytest.mark.parametrize("task,filename,sim_port,inference_port", [
    ("table_buttons", "table_buttons_parallel_vlajepa_eval.json", 18421, 18451),
    ("playground", "playground_parallel_walloss_eval.json", 18422, 18452),
])
def test_parallel_examples_keep_camera_calibration_and_matching_env_counts(task, filename, sim_port, inference_port):
    """Compile both node sides and resolve task inputs without starting Isaac."""
    import importlib
    compiled = load_experiment(PROJECT_ROOT / "configs/experiments" / filename)
    nodes = compiled["nodes"]
    simulation, inference, trainer = (nodes[key] for key in ("simulation", "inference", "trainer"))
    assert compiled["factories"]["task"] == resolve("task", task)
    config = importlib.import_module(f"pressb.tasks.{task}.backend").load_config(simulation["backend"]["config"])
    assert config["num_envs"] == trainer["rollout"]["num_envs"] == 8
    assert inference["service"]["batch_size"] == 8
    assert config["render_resolution"] == [640, 480]
    assert config["horizon"] == trainer["rollout"]["execution_horizon"]
    assert trainer["endpoints"]["simulation"] == f"http://127.0.0.1:{sim_port}"
    assert trainer["endpoints"]["inference"] == f"ws://127.0.0.1:{inference_port}"
    scene = json.loads(Path(config["scene_config"]).read_text())
    camera_fields = {key: value for key, value in scene.items() if "camera" in key}
    assert "up_camera" in camera_fields
    assert any("wrist_camera" in key for key in camera_fields)
    assert {key: config["scene"][key] for key in camera_fields} == camera_fields
    if task == "table_buttons":
        assert config["action_dim"] == 6
        assert config["randomization"] == {"profiles": ["medium"], "colors": ["red", "yellow", "blue"], "positions": True}
    else:
        assert not config["scene"]["objects"]
        assert config["scene"]["static_props"] == scene["static_props"]


@pytest.mark.parametrize("kind,name", [(k, n) for k, rows in available().items() for n in rows])
def test_resolved_factory_exists_without_initializing(kind, name):
    import importlib
    module, function = resolve(kind, name).split(":")
    assert callable(getattr(importlib.import_module(module), function))


def test_inheritance_paths_lists_and_endpoints(tmp_path):
    folder = tmp_path / "parents"
    folder.mkdir()
    base = minimal()
    base["task"]["config"] = {"x": "${experiment_dir}/a.json", "list": [1, 2]}
    base["nodes"] = {"simulation": {"server": {"host": "0.0.0.0", "port": 19001}}}
    save(folder / "base.json", base)
    p = save(tmp_path / "child.json", dict(extends="parents/base.json", name="child",
        task={"config": {"list": [3]}}, run={"endpoints": {"inference": "ws://other:1234"}}))
    result = load_experiment(p)
    assert result["nodes"]["simulation"]["backend"]["config"] == {"x": str(folder / "a.json"), "list": [3]}
    assert result["nodes"]["trainer"]["endpoints"] == {"inference": "ws://other:1234", "simulation": "http://127.0.0.1:19001"}
    assert len(result["sources"]) == 2


def test_cycle_and_missing_variable_fail(tmp_path):
    p = save(tmp_path / "a.json", {"extends": "b.json"})
    save(tmp_path / "b.json", {"extends": "a.json"})
    with pytest.raises(ValueError, match="cycle"):
        load_experiment(p)
    d = minimal(); d["name"] = "${not_defined}"
    with pytest.raises(ValueError, match="Unresolved"):
        load_experiment(save(p, d))


@pytest.mark.parametrize("change", [
    {"schema_version": True}, {"name": "../bad"}, {"task": {"plugin": "typo"}},
    {"model": {"plugin": "walloss", "factory": "a:b"}}, {"model": {"factory": "invalid"}},
    {"nodes": {"invalid": {}}}, {"nodes": {"simulation": {"server": {"port": True}}}},
    {"collection": {"include_images": 1}}, {"run": {"endpoints": {"token": "secret"}}},
])
def test_invalid_config_rejected(tmp_path, change):
    with pytest.raises(ValueError):
        load_experiment(save(tmp_path / "bad.json", {**minimal(), **change}))


def test_prepare_is_idempotent_and_refuses_different_config(tmp_path):
    result = load_experiment(save(tmp_path / "input.json", minimal()))
    out = tmp_path / "resolved"
    paths = materialize(result, out)
    assert materialize(result, out) == paths
    result["nodes"]["simulation"]["server"]["port"] += 1
    with pytest.raises(FileExistsError):
        materialize(result, out)
    assert json.loads(paths["simulation"].read_text())["server"]["port"] == 8101


@pytest.mark.parametrize("path", sorted((PROJECT_ROOT/"configs/experiments").glob("*.json")), ids=lambda p: p.stem)
def test_shipped_experiments_validate(path, capsys):
    assert main(["validate", "--config", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True


def test_service_cli_does_not_import_trainer_or_models(tmp_path):
    # Prepare/validate are CPU commands; actual service startup must remain lazy.
    script = "import pressb.experiments.cli,sys; assert 'pressb.runtime.trainer' not in sys.modules; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", script], check=True,
                   env={**__import__('os').environ, "PYTHONPATH": str(PROJECT_ROOT/'src')})


def test_cli_collection_uses_shared_rollout_and_separate_dataset(tmp_path, monkeypatch, capsys):
    import torch
    import pressb.runtime.trainer as trainer
    from test_rl_nodes_trainer import config, metadata, Inference, Simulation
    torch.set_num_threads(1)
    inf, sim = metadata()
    clients = {"ws://127.0.0.1:8100": Inference(inf), "http://127.0.0.1:8101": Simulation(sim)}
    monkeypatch.setattr(trainer, "NodeClient", lambda endpoint, **kw: clients[endpoint])
    data = minimal()
    data["run"] = config(tmp_path / "run", method="base", steps=20)
    data["run"]["mode"] = "eval"
    path = save(tmp_path / "config.json", data)
    assert main(["collect", "--config", str(path), "--episodes", "2", "--resolved-dir", str(tmp_path / "resolved")]) == 0
    manifest = json.loads((tmp_path / "run_trajectories/manifest.json").read_text())
    assert sum(ep["complete"] for ep in manifest["episodes"].values()) == 2
    assert json.loads(capsys.readouterr().out)["state"] == "completed"
    assert json.loads((tmp_path/"run/config.json").read_text())["rollout"]["terminal_observation_mode"] == "full"


def test_explicit_dataset_path_is_never_silently_relocated(tmp_path):
    data = minimal()
    data["run"]["output_dir"] = str(tmp_path / "run")
    data["collection"] = {"directory": str(tmp_path / "run/data")}
    path = save(tmp_path / "config.json", data)
    with pytest.raises(ValueError, match="outside"):
        main(["collect", "--config", str(path), "--resolved-dir", str(tmp_path / "resolved")])
    assert not (tmp_path / "run").exists()
    assert not (tmp_path / "run_trajectories").exists()
