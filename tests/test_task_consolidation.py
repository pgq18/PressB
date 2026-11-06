"""Regression boundaries for task consolidation and historical imports."""
import ast
import importlib
import pickle
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TABLE_MODULES = (
    "camera_domain", "collection", "collision", "execution", "motion",
    "motion_templates", "placement", "policy_evaluation", "spacing", "task",
    "target_relative_motion", "workspace",
)
ALIASES = [
    ("pressb.pick_place.task", "pressb.tasks.pap.task"),
    ("pressb.pick_place.evaluation", "pressb.tasks.pap.evaluation"),
    ("pressb.pick_place.wallx", "pressb.compat.wallx"),
    ("pressb.table_buttons.policy_evaluation_joint6", "pressb.compat.table_buttons_policy"),
] + [("pressb.table_buttons." + name, "pressb.tasks.table_buttons." + name)
     for name in TABLE_MODULES]


@pytest.mark.parametrize("legacy,current", ALIASES)
def test_old_task_imports_share_canonical_module_and_monkeypatches(legacy, current, monkeypatch):
    old = importlib.import_module(legacy)
    new = importlib.import_module(current)
    assert old is new
    marker = object()
    monkeypatch.setattr(old, "_consolidation_probe", marker, raising=False)
    assert new._consolidation_probe is marker


@pytest.mark.parametrize("legacy,current", [
    ("pressb.pick_place.scene", "pressb.tasks.pap.scene"),
    ("pressb.table_buttons.scene", "pressb.tasks.table_buttons.scene"),
])
def test_old_scene_imports_share_the_live_builder(legacy, current):
    pytest.importorskip("pxr.Usd")
    old, new = importlib.import_module(legacy), importlib.import_module(current)
    assert old is new
    assert old.build_scene is new.build_scene


@pytest.mark.parametrize("module,name", [
    ("pressb.pick_place.task", "PickPlaceOutcome"),
    ("pressb.table_buttons.task", "TableButtonOutcome"),
    ("pressb.table_buttons.policy_evaluation", "InvalidPolicyAction"),
])
def test_saved_old_class_paths_remain_loadable(module, name, monkeypatch):
    cls = getattr(importlib.import_module(module), name)
    with monkeypatch.context() as patch:
        patch.setattr(cls, "__module__", module)
        serialized = pickle.dumps(cls, protocol=4)
    assert module.encode() in serialized
    assert pickle.loads(serialized) is cls


def test_initialization_and_legacy_clients_have_one_implementation():
    from pressb.compat import wallx, table_buttons_policy
    from pressb.tasks.pap.initialization import load_policy_initial
    from pressb.tasks.table_buttons import policy_evaluation
    assert wallx.load_policy_initial is load_policy_initial
    assert policy_evaluation.HTTPJointPolicy is table_buttons_policy.HTTPJointPolicy
    assert policy_evaluation.png_base64 is table_buttons_policy.png_base64
    assert policy_evaluation.InvalidPolicyAction is table_buttons_policy.InvalidPolicyAction


def _absolute_imports(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.level:
                relative = path.relative_to(ROOT / "src").with_suffix("")
                package = ".".join(relative.parts[:-1])
                yield importlib.util.resolve_name("." * node.level + node.module, package)
            else:
                yield node.module


def test_current_runtime_tasks_and_scripts_do_not_depend_on_old_task_packages():
    paths = [*ROOT.joinpath("src/pressb/tasks").rglob("*.py"),
             *ROOT.joinpath("src/pressb/runtime").glob("*.py"),
             *ROOT.joinpath("scripts").glob("*.py")]
    forbidden = ("pressb.pick_place", "pressb.table_buttons")
    leaks = [(str(path.relative_to(ROOT)), name) for path in paths for name in _absolute_imports(path)
             if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)]
    assert not leaks


def test_shared_scene_and_button_scene_do_not_import_pap_implementation():
    shared = [ROOT / "src/pressb/runtime" / name for name in ("scene.py", "workbench.py")]
    button = ROOT / "src/pressb/tasks/table_buttons/scene.py"
    for path in [*shared, button]:
        imports = list(_absolute_imports(path))
        assert not any(name.startswith(("pressb.tasks.pap", "pressb.pick_place")) for name in imports)
        if path in shared:
            assert not any(name.startswith("pressb.tasks.") for name in imports)


def test_collector_archives_canonical_scene_and_physics_dependencies():
    path = ROOT / "scripts/run_table_buttons_collection_suite.py"
    spec = importlib.util.spec_from_file_location("_consolidation_collection", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = {str(p.relative_to(ROOT)) for p in module.frozen_paths()}
    for source in ("tasks/table_buttons/scene.py", "tasks/table_buttons/task.py",
                   "tasks/table_buttons/target_relative_motion.py", "runtime/scene.py",
                   "runtime/workbench.py", "runtime/isaac.py", "runtime/recording.py"):
        assert "src/pressb/" + source in paths
