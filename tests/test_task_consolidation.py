"""Canonical task ownership, removed package boundaries and archive coverage."""
import ast
import hashlib
import importlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REMOVED_PACKAGES = ("pressb.pick_place", "pressb.table_buttons")


@pytest.mark.parametrize("module", REMOVED_PACKAGES)
def test_removed_task_packages_have_no_directory_or_import(module):
    path = ROOT / "src" / module.replace(".", "/")
    assert not path.exists()
    assert not path.with_suffix(".py").exists()
    assert importlib.util.find_spec(module) is None
    with pytest.raises(ModuleNotFoundError) as error:
        importlib.import_module(module)
    assert error.value.name == module


@pytest.mark.parametrize("module,name", [
    ("pressb.tasks.pap.task", "PickPlaceOutcome"),
    ("pressb.tasks.pap.initialization", "load_policy_initial"),
    ("pressb.tasks.table_buttons.task", "TableButtonOutcome"),
    ("pressb.tasks.table_buttons.policy_contract", "InvalidPolicyAction"),
    ("pressb.compat.wallx", "WallXClient"),
    ("pressb.compat.table_buttons_policy", "HTTPJointPolicy"),
    ("pressb.compat.table_buttons_policy", "HTTPJoint6Policy"),
])
def test_public_task_and_client_symbols_belong_to_canonical_files(module, name):
    implementation = importlib.import_module(module)
    assert implementation.__name__ == module
    assert Path(implementation.__file__) == ROOT / "src" / (module.replace(".", "/") + ".py")
    assert getattr(implementation, name).__module__ == module


def test_initialization_and_compat_clients_have_one_implementation():
    from pressb.compat import wallx, table_buttons_policy
    from pressb.tasks.pap.initialization import load_policy_initial
    from pressb.tasks.table_buttons import policy_contract, policy_evaluation
    assert wallx.load_policy_initial is load_policy_initial
    assert policy_evaluation.HTTPJointPolicy is table_buttons_policy.HTTPJointPolicy
    assert policy_evaluation.png_base64 is table_buttons_policy.png_base64
    assert policy_evaluation.InvalidPolicyAction is policy_contract.InvalidPolicyAction
    assert table_buttons_policy.InvalidPolicyAction is policy_contract.InvalidPolicyAction
    assert issubclass(table_buttons_policy.HTTPJoint6Policy, table_buttons_policy.HTTPJointPolicy)


def _absolute_imports(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                relative = path.relative_to(ROOT / "src").with_suffix("")
                package = ".".join(relative.parts[:-1])
                module = importlib.util.resolve_name("." * node.level + module, package)
            yield module
            yield from (module + "." + alias.name for alias in node.names)
        elif isinstance(node, ast.Call) and node.args:
            # Include literal dynamic imports used by entry points and plugins.
            function = node.func
            name = function.id if isinstance(function, ast.Name) else getattr(function, "attr", None)
            if name in ("import_module", "__import__"):
                argument = node.args[0]
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    yield argument.value


def test_sources_scripts_and_tests_do_not_import_removed_task_packages():
    paths = [path for directory in ("src", "scripts", "tests")
             for path in ROOT.joinpath(directory).rglob("*.py")]
    leaks = [(str(path.relative_to(ROOT)), name) for path in paths for name in _absolute_imports(path)
             if any(name == prefix or name.startswith(prefix + ".") for prefix in REMOVED_PACKAGES)]
    assert not leaks


def test_shared_scene_has_no_task_implementation_dependency():
    shared = [ROOT / "src/pressb/runtime" / name for name in ("scene.py", "workbench.py")]
    for path in shared:
        assert not any(name == "pressb.tasks" or name.startswith("pressb.tasks.")
                       for name in _absolute_imports(path))


@pytest.mark.parametrize("task,other", [("pap", "table_buttons"), ("table_buttons", "pap")])
def test_task_implementations_do_not_depend_on_each_other(task, other):
    prefix = "pressb.tasks." + other
    paths = ROOT.joinpath("src/pressb/tasks", task).rglob("*.py")
    leaks = [(str(path.relative_to(ROOT)), name) for path in paths for name in _absolute_imports(path)
             if name == prefix or name.startswith(prefix + ".")]
    assert not leaks


def test_collector_archives_canonical_scene_and_physics_dependencies(tmp_path):
    path = ROOT / "scripts/run_table_buttons_collection_suite.py"
    spec = importlib.util.spec_from_file_location("_consolidation_collection", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = {str(p.relative_to(ROOT)) for p in module.frozen_paths()}
    for source in ("tasks/table_buttons/scene.py", "tasks/table_buttons/task.py",
                   "tasks/table_buttons/target_relative_motion.py", "runtime/scene.py",
                   "runtime/workbench.py", "runtime/isaac.py", "runtime/recording.py"):
        assert "src/pressb/" + source in paths
    assert {str(path.relative_to(ROOT))
            for path in ROOT.joinpath("src/pressb/tasks/table_buttons").glob("*.py")} <= paths
    assert not any(path.startswith(("src/pressb/pick_place/", "src/pressb/table_buttons/"))
                   for path in paths)
    module.freeze_code(tmp_path)
    manifest = json.loads((tmp_path / "source_archive/manifest.json").read_text())
    assert set(manifest["files"]) == paths
    for path, identity in manifest["files"].items():
        contents = (ROOT / path).read_bytes()
        assert (tmp_path / "source_archive" / path).read_bytes() == contents
        assert identity == {"sha256": hashlib.sha256(contents).hexdigest(), "bytes": len(contents)}
    module.verify_code(tmp_path)
