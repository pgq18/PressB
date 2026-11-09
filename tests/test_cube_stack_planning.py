"""CPU geometry, bounded IK and physical-outcome audit checks for stacking."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pressb.robots.piper.kinematics import PiperKinematics
from pressb.tasks.cube_stack.planning import plan_stack

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/"scripts/smoke_cube_stack.py"
spec = importlib.util.spec_from_file_location("cube_stack_smoke",SCRIPT)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


@pytest.fixture(scope="module")
def scene():
    cfg = json.loads((ROOT/"configs/piper_playground.json").read_text())
    cfg["objects"] = [dict(name="red",center_xy=[.331263,.102522],size=[.029]*3,yaw_deg=-10.51),
        dict(name="blue",center_xy=[.326851,-.083409],size=[.030]*3,yaw_deg=30.36)]
    return cfg


@pytest.fixture(scope="module")
def plan(scene):
    return plan_stack(scene)


def test_physical_plan_lifts_and_rotates_before_placing_then_returns_home(scene,plan):
    phases = list(dict.fromkeys(plan["phases"]))
    assert phases == ["open","pregrasp","descend","close","lift","transfer","place","release","retreat","return_home","settled"]
    actions = np.asarray(plan["actions"])
    assert actions.shape[1] == 7 and plan["hardware_access"] is False
    assert plan["transfer_yaw_rotation_deg"] == pytest.approx(40.87)
    np.testing.assert_allclose(actions[-1,:6],scene["initial_state_joint_rad_m"][:6])
    assert actions[-1,6] == pytest.approx(.06)
    kin = PiperKinematics(ROOT/scene["robot_urdf"],tip_offset=.1358)
    placed = next(row for row in reversed(plan["waypoints"]) if row["phase"]=="place")
    transform = kin.fk(np.asarray(placed["action"][:6]))
    np.testing.assert_allclose(transform[:3,3],[.326851,-.083409,.012+.030+.001-.006],atol=1e-5)
    expected = Rotation.from_euler("z",30.36+180,degrees=True).as_matrix()@np.diag([1,-1,-1])
    np.testing.assert_allclose(transform[:3,:3],expected,atol=1e-5)
    assert plan["min_tcp_base_z_m"] >= .006-1e-5
    assert max(plan["measured_max_speed_rad_m_s"][:6]) <= np.deg2rad(20)+1e-8


@pytest.mark.parametrize("kwargs",[{"close_width":.031},{"lift_height_above_table":.05},
    {"grasp_height_above_table":.04},{"cartesian_speed_m_s":float("nan")}])
def test_impossible_grasp_parameters_fail_before_any_simulator(scene,kwargs):
    with pytest.raises(ValueError):
        plan_stack(scene,**kwargs)


def history(plan):
    rows = []
    for index,z in enumerate([.7745,.86]+[.8045]*20):
        rows.append(dict(objects={"red":{"position_world":[.3,.1,z],"orientation_wxyz":[1,0,0,0]},
            "blue":{"position_world":[.3,.1,.775],"orientation_wxyz":[1,0,0,0]}},
            state=[*plan["expected_initial_state"][:6],.06],info={"success":True},
            phase="settled",action_index=len(plan["actions"])-1))
    return rows


def test_final_audit_needs_measured_lift_release_stable_stack_and_completed_return(plan):
    values = history(plan)
    result = smoke.assess_execution(values,plan)
    assert result["passed"] and result["center_height_separation_error_m"] < 1e-9
    changed = deepcopy(values)
    changed[-1]["info"]["success"] = False
    assert not smoke.assess_execution(changed,plan)["passed"]
    changed = deepcopy(values)
    changed[-1]["objects"]["red"]["position_world"][2] += .006
    assert not smoke.assess_execution(changed,plan)["passed"]
    changed = deepcopy(values)
    changed[-1]["state"][6] = .027
    assert not smoke.assess_execution(changed,plan)["passed"]
    changed = deepcopy(values)
    changed[-1]["phase"] = "release"
    assert not smoke.assess_execution(changed,plan)["passed"]


def test_help_imports_neither_task_runtime_nor_hardware():
    code = '''
import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('pressb','isaacsim','omni','carb','pxr','can','pyrealsense2'):
            raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
sys.argv=[sys.argv[1],'--help']
runpy.run_path(sys.argv[0],run_name='__main__')
'''
    result = subprocess.run([sys.executable,"-c",code,str(SCRIPT)],capture_output=True,text=True)
    assert result.returncode == 0,result.stderr
    assert "--plan-only" in result.stdout


def test_actual_plan_only_uses_pinned_scene_and_disables_random_layouts(tmp_path):
    args = smoke.parser().parse_args(["--plan-only","--output",str(tmp_path/"plan")])
    assert smoke.run(args) == 0
    config = json.loads((args.output/"config.json").read_text())
    generated = json.loads((args.output/"plan.json").read_text())
    report = json.loads((args.output/"report.json").read_text())
    assert config["scene_config"] == "configs/calibration/playground_20261109_global_refined.json"
    assert config["randomization"]["enabled"] is False
    assert config["terminate_on_success"] is False
    assert generated["source"]["center_xy"] == [.331263,.102522]
    assert generated["target"]["center_xy"] == [.326851,-.083409]
    assert generated["source"]["yaw_deg"] == -10.51
    assert generated["target"]["yaw_deg"] == 30.36
    assert report["status"] == "planned" and report["physical_execution"] is False
