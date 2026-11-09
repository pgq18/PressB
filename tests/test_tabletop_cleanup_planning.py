"""CPU reachability and end-to-end audit requirements for cleanup demos."""
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
from pressb.tasks.tabletop_cleanup.geometry import asset_spec
from pressb.tasks.tabletop_cleanup.planning import placement_slots,plan_cleanup

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"scripts/smoke_tabletop_cleanup.py"
spec=importlib.util.spec_from_file_location("cleanup_smoke",SCRIPT)
smoke=importlib.util.module_from_spec(spec);spec.loader.exec_module(smoke)


@pytest.fixture(scope="module")
def config():
    # Two present props/box use the fresh registered layout. Lidar is absent
    # there; its reserved CPU-test slot is synthetic and not a live observation.
    return dict(scene=json.loads((ROOT/"configs/calibration/playground_20261109_global_refined.json").read_text()),
        objects={"speaker":dict(center_xy=[.371758,-.10155661],yaw_deg=1.35020),
                 "lidar":dict(center_xy=[.3552321,-.1871806],yaw_deg=.52981),
                 "gamepad":dict(center_xy=[.36584942,.03786663],yaw_deg=5.66315)},
        bin=dict(center_xy=[.33666862,.22993637],yaw_deg=1.10466,
                 size_xyz_m=[.222,.148,.049],wall_thickness_m=.004))


@pytest.fixture(scope="module")
def plan(config):
    return plan_cleanup(config)


@pytest.mark.parametrize("kinds",[("speaker","lidar"),("gamepad","lidar"),("gamepad","speaker")])
def test_full_visual_envelopes_fit_two_distinct_bin_slots(config,kinds):
    slots=placement_slots(config,kinds)
    box=config["bin"]
    r=Rotation.from_euler("z",box["yaw_deg"],degrees=True).as_matrix()[:2,:2]
    intervals=[]
    for kind in kinds:
        # All appendages count: speaker ears, gamepad grips and thumbsticks.
        vertices=np.asarray(asset_spec(kind)["support_vertices"])
        local=(np.asarray(slots[kind]["center_xy"])-box["center_xy"])@r
        points=vertices[:,:2]+local
        inner=np.asarray(box["size_xyz_m"][:2])/2-box["wall_thickness_m"]
        assert np.all(points>=-inner+.003-1e-9) and np.all(points<=inner-.003+1e-9)
        intervals.append((points[:,0].min(),points[:,0].max()))
    left,right=sorted(intervals)
    assert right[0]-left[1]>=.006-1e-9


@pytest.mark.parametrize("kinds",[("speaker","lidar"),("gamepad","lidar"),("gamepad","speaker")])
def test_review_layout_has_complete_bounded_plan_for_each_pair(config,kinds):
    generated=plan_cleanup(config,kinds=kinds)
    actions=np.asarray(generated["actions"])
    kin=PiperKinematics(ROOT/config["scene"]["robot_urdf"],tip_offset=.1358)
    assert generated["hardware_access"] is False and generated["simulation_only"] is True
    assert actions.shape[1]==7
    assert np.all(actions[:,:6]>=kin.lower-1e-9) and np.all(actions[:,:6]<=kin.upper+1e-9)
    assert max(generated["measured_max_speed_rad_m_s"][:6])<=np.deg2rad(20)+1e-8
    assert generated["measured_max_speed_rad_m_s"][6]<=.030+1e-8
    np.testing.assert_allclose(actions[-1,:6],config["scene"]["initial_state_joint_rad_m"][:6])
    phases=list(dict.fromkeys(generated["phases"]))
    expected=[]
    for kind in kinds:
        expected += [kind+":"+phase for phase in ("open","pregrasp","descend","close","lift",
            "orient_carry","transfer","place","release","retreat","return_home")]
        for phase,key in (("descend","grasp_tcp_base_m"),("place","placement_tcp_base_m")):
            waypoint=next(row for row in reversed(generated["waypoints"]) if row["phase"]==kind+":"+phase)
            fk=kin.fk(np.asarray(waypoint["action"][:6]))
            np.testing.assert_allclose(fk[:3,3],generated["grasps"][kind][key],atol=5e-4)
            assert fk[2,2]<-.9999
    assert phases==expected+["settled"]


@pytest.mark.parametrize("kwargs",[dict(kinds=["speaker","speaker"]),dict(kinds=["speaker"]),
    dict(cartesian_speed_m_s=float("nan")),dict(lift_height_above_table_m=.05),dict(contact_to_tip_m=.020)])
def test_invalid_or_colliding_requests_fail_before_simulator(config,kwargs):
    with pytest.raises(ValueError):plan_cleanup(config,**kwargs)


def test_optional_wider_approach_opening_preserves_close_contact_and_defaults(config,plan):
    widened=plan_cleanup(config,grasp_openings_m={"speaker":.070})
    assert plan["grasps"]["speaker"]["open_width_m"]==pytest.approx(.052)
    assert widened["grasps"]["speaker"]["open_width_m"]==pytest.approx(.070)
    assert widened["grasps"]["speaker"]["close_width_m"]==plan["grasps"]["speaker"]["close_width_m"]
    np.testing.assert_allclose(widened["grasps"]["speaker"]["grasp_tcp_base_m"],plan["grasps"]["speaker"]["grasp_tcp_base_m"])
    assert widened["grasp_opening_overrides_m"]=={"speaker":.070}
    for bad in ({"speaker":.080},{"speaker":.038},{"gamepad":.070},{"speaker":float("nan")}):
        with pytest.raises(ValueError):plan_cleanup(config,grasp_openings_m=bad)


def test_taller_gamepad_old_far_release_remains_rejected_at_original_ik_limits(config):
    old=deepcopy(config)
    old["bin"].update(center_xy=[.3441528,.2327083],yaw_deg=-3.3725)
    # The old thin-shell demo used a lower TCP release height. Corrected
    # geometry puts this endpoint past J5 reach; the solver must reject it.
    with pytest.raises(ValueError,match="Unreachable PiPER tool target"):
        plan_cleanup(old,kinds=("gamepad","speaker"))


def _history(plan):
    rows=[]
    for index,z in enumerate([.77,.86]+[.774]*20):
        objects={kind:dict(position_world=[.35,.23,z],orientation_wxyz=[1,0,0,0]) for kind in plan["kinds"]}
        flags={kind:dict.fromkeys(("active","grasped_and_lifted","contained","supported","released","low_velocity"),True)
               for kind in plan["kinds"]}
        rows.append(dict(objects=objects,state=[*plan["expected_initial_state"][:6],.060],
            info=dict(success=True,physical_outcome=dict(success=True,objects=flags)),phase="settled",
            action_index=len(plan["actions"])-1))
    return rows


def test_audit_requires_both_physical_grasps_release_stability_and_return(plan):
    history=_history(plan)
    assert smoke.assess_execution(history,plan)["passed"]
    for key in ("grasped_and_lifted","contained","supported","released"):
        changed=deepcopy(history)
        changed[-1]["info"]["physical_outcome"]["objects"]["lidar"][key]=False
        assert not smoke.assess_execution(changed,plan)["passed"]
    changed=deepcopy(history)
    for row in changed:row["objects"]["lidar"]["position_world"][2]=.774
    assert not smoke.assess_execution(changed,plan)["passed"]
    changed=deepcopy(history);changed[-1]["phase"]="lidar:release"
    assert not smoke.assess_execution(changed,plan)["passed"]
    changed=deepcopy(history);changed[-1]["state"][6]=.035
    assert not smoke.assess_execution(changed,plan)["passed"]


def test_help_does_not_import_simulation_or_hardware():
    code='''
import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('pressb','isaacsim','omni','carb','pxr','can','pyrealsense2'):
            raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
sys.argv=[sys.argv[1],'--help']
runpy.run_path(sys.argv[0],run_name='__main__')
'''
    result=subprocess.run([sys.executable,"-c",code,str(SCRIPT)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert "--plan-only" in result.stdout


def test_plan_only_uses_real_layout_and_pins_exact_two_without_isaac(tmp_path):
    args=smoke.parser().parse_args(["--plan-only","--output",str(tmp_path/"plan")])
    assert smoke.run(args)==0
    config=json.loads((args.output/"config.json").read_text())
    generated=json.loads((args.output/"plan.json").read_text())
    report=json.loads((args.output/"report.json").read_text())
    assert config["randomization"]["enabled"] is False
    assert set(config["randomization"]["fixed_subset"])=={"gamepad","speaker"}
    assert config["horizon"]==1 and config["terminate_on_success"] is False
    assert config["scene_config"]=="configs/calibration/playground_20261109_global_refined.json"
    assert generated["kinds"]==["gamepad","speaker"]
    canonical=json.loads((ROOT/"configs/rl_nodes/tabletop_cleanup_real_layout.json").read_text())["backend"]["config"]
    assert generated["bin"]==canonical["bin"]
    np.testing.assert_allclose(generated["expected_initial_state"],[-.18580775216731635,.4724955350999049,
        -.45820128852607134,-.11393509357018983,.7485193563028081,.08302531251737025,.0595])
    assert report["status"]=="planned" and report["physical_execution"] is False
