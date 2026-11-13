"""CPU reachability and end-to-end audit requirements for cleanup demos."""
from copy import deepcopy
from itertools import permutations
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
    return dict(scene=json.loads((ROOT/"configs/calibration/playground_20261113_appearance.json").read_text()),
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


@pytest.mark.parametrize("kinds",list(permutations(("speaker","lidar","gamepad"),2)))
def test_review_layout_has_complete_bounded_plan_for_each_order(config,kinds):
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
            "orient_carry","transfer","place","release","retreat")]
        for phase,key in (("descend","grasp_tcp_base_m"),("place","placement_tcp_base_m")):
            waypoint=next(row for row in reversed(generated["waypoints"]) if row["phase"]==kind+":"+phase)
            fk=kin.fk(np.asarray(waypoint["action"][:6]))
            np.testing.assert_allclose(fk[:3,3],generated["grasps"][kind][key],atol=5e-4)
            assert fk[2,2]<-.9999
    assert phases==expected+[kinds[-1]+":return_home","settled"]
    _assert_direct_second_pickup(generated)


def _assert_direct_second_pickup(generated):
    """The second approach starts at the previous retreat, not a home detour."""
    first,second=generated["kinds"]
    waypoints=generated["waypoints"]
    actions=np.asarray(generated["actions"])
    initial=np.asarray(generated["expected_initial_state"])
    home=[w for w in waypoints if w["phase"].endswith(":return_home")]
    assert len(home)==1 and home[0]["phase"]==second+":return_home"
    retreat=next(w for w in reversed(waypoints) if w["phase"]==first+":retreat")
    opening=next(w for w in waypoints if w["phase"]==second+":open")
    approach=next(w for w in reversed(waypoints) if w["phase"]==second+":pregrasp")
    # Changing the gripper at the handover must hold the preceding arm pose.
    np.testing.assert_allclose(opening["action"][:6],retreat["action"][:6],atol=1e-12,rtol=0)
    assert opening["start_frame"]==retreat["end_frame"]+1
    transition=actions[retreat["end_frame"]:approach["end_frame"]+1,:6]
    assert len(transition)>1
    assert np.min(np.max(np.abs(transition-initial[:6]),axis=1))>np.deg2rad(1)
    # Check the actual samples, including phase boundaries, independently of
    # the planner's reported maximum speed or its waypoint interpolation.
    velocity=np.abs(np.diff(np.vstack([initial,actions]),axis=0))*generated["fps"]
    assert np.max(velocity[:,:6])<=np.deg2rad(generated["max_joint_speed_deg_s"])+1e-8
    assert np.max(velocity[:,6])<=.030+1e-8
    np.testing.assert_allclose(actions[-1,:6],initial[:6],atol=1e-12,rtol=0)


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
    for bad in ({"speaker":.101},{"speaker":.038},{"gamepad":.070},{"speaker":float("nan")}):
        with pytest.raises(ValueError):plan_cleanup(config,grasp_openings_m=bad)


@pytest.mark.parametrize("kinds",list(permutations(("speaker","lidar","gamepad"),2)))
def test_compact_plan_is_continuous_full_open_and_has_no_wait_or_home(config,kinds):
    generated=plan_cleanup(config,kinds=kinds,trajectory_profile="compact")
    actions=np.asarray(generated["actions"])
    initial=np.asarray(generated["expected_initial_state"])
    kin=PiperKinematics(ROOT/config["scene"]["robot_urdf"],tip_offset=.1358)
    assert generated["trajectory_profile"]=="compact"
    assert generated["return_home_policy"]=="none"
    assert generated["validation_hold_seconds"]==pytest.approx(.6)
    assert np.all(actions[:,:6]>=kin.lower-1e-9) and np.all(actions[:,:6]<=kin.upper+1e-9)
    delta=np.abs(np.diff(np.vstack([initial,actions]),axis=0))
    assert np.all(np.max(delta,axis=1)>1e-10)
    assert np.max(delta[:,:6])*generated["fps"]<=np.deg2rad(20)+1e-8
    assert np.max(delta[:,6])*generated["fps"]<=.030+1e-8
    assert all(w["hold_seconds"]==0 for w in generated["waypoints"])
    expected=[k+":"+phase for k in kinds for phase in
              ("pregrasp","descend","close","lift","transfer","place","release","retreat")]
    assert list(dict.fromkeys(generated["phases"]))==expected
    for kind in kinds:
        grasp=generated["grasps"][kind]
        assert grasp["open_width_m"]==pytest.approx(.100)
        assert grasp["lift_reorientation"]["blended_during_lift"]
        assert grasp["close_width_m"]==asset_spec(kind)["grasp_candidates"][0]["closed_width_m"]
    # Full opening takes place concurrently with the first approach.
    first=generated["waypoints"][0]
    assert first["phase"]==kinds[0]+":pregrasp"
    assert first["action"][6]==pytest.approx(.100)
    assert not np.allclose(first["action"][:6],initial[:6])
    # The second approach starts immediately after the bin withdrawal.
    retreat=next(w for w in reversed(generated["waypoints"]) if w["phase"]==kinds[0]+":retreat")
    approach=next(w for w in generated["waypoints"] if w["phase"]==kinds[1]+":pregrasp")
    assert approach["start_frame"]==retreat["end_frame"]+1
    transition=actions[retreat["end_frame"]:approach["end_frame"]+1,:6]
    assert np.min(np.max(np.abs(transition-initial[:6]),axis=1))>np.deg2rad(1)
    assert not np.allclose(actions[-1,:6],initial[:6])


def test_compact_shortens_execution_without_deleting_interior_samples(config):
    baseline=plan_cleanup(config,kinds=("gamepad","speaker"),
                          grasp_openings_m={"gamepad":.1,"speaker":.1})
    compact=plan_cleanup(config,kinds=("gamepad","speaker"),trajectory_profile="compact")
    assert compact["fps"]==baseline["fps"]
    assert compact["duration_s"]<baseline["duration_s"]-10.
    assert compact["duration_s"]==len(compact["actions"])/compact["fps"]
    assert len(compact["phases"])==len(compact["actions"])
    # No recording mask or time compression: every planned frame is executed.
    assert "record_mask" not in compact


def test_compact_lift_keeps_settled_gamepad_sweep_above_table(config):
    orientation=Rotation.from_euler("xyz",[-9.6,0.,5.6757],degrees=True)
    pose=dict(position_base_m=[.36536324,.04280226,.02307460],
              orientation_wxyz=orientation.as_quat()[[3,0,1,2]].tolist())
    generated=plan_cleanup(config,kinds=("gamepad","speaker"),
                           object_poses_base={"gamepad":pose},trajectory_profile="compact")
    grasp=generated["grasps"]["gamepad"]
    assert grasp["finger_sweep_clearance"]["opening_m"]==pytest.approx(.1)
    clearances=grasp["finger_sweep_clearance"]["after_clearance_m"]
    assert min(clearances.values())>.010
    vertices=np.asarray(asset_spec("gamepad")["support_vertices"])
    held=((vertices@orientation.as_matrix().T+pose["position_base_m"])
          -grasp["grasp_tcp_base_m"])@np.asarray(grasp["grasp_rotation_base"])
    kin=PiperKinematics(ROOT/config["scene"]["robot_urdf"],tip_offset=.1358)
    mount=config["scene"]["robot_base_height_above_table_m"]
    rotated_and_lifted=[]
    for action,phase in zip(generated["actions"],generated["phases"]):
        if phase!="gamepad:lift":
            continue
        transform=kin.fk(np.asarray(action[:6]))
        if np.max(np.abs(transform[:3,:3]-grasp["grasp_rotation_base"]))>.001:
            points=held@transform[:3,:3].T+transform[:3,3]
            assert np.min(points[:,2])+mount>.004
            rotated_and_lifted.append(transform[:3,3])
    assert len(rotated_and_lifted)>2
    assert np.ptp(np.asarray(rotated_and_lifted)[:,2])>.010


def test_unknown_trajectory_profile_is_rejected(config):
    with pytest.raises(ValueError,match="trajectory_profile"):
        plan_cleanup(config,trajectory_profile="fastest")


def test_taller_gamepad_old_far_release_remains_rejected_at_original_ik_limits(config):
    old=deepcopy(config)
    old["bin"].update(center_xy=[.3441528,.2327083],yaw_deg=-3.3725)
    # The old thin-shell demo used a lower TCP release height. Corrected
    # geometry puts this endpoint past J5 reach; the solver must reject it.
    with pytest.raises(ValueError,match="Unreachable PiPER tool target"):
        plan_cleanup(old,kinds=("gamepad","speaker"))


def test_measured_tilted_gamepad_approaches_along_surface_and_places_upright(config):
    original=deepcopy(config)
    orientation=Rotation.from_euler("xyz",[-9.6,0.,5.6757],degrees=True)
    quat=orientation.as_quat()[[3,0,1,2]]
    pose=dict(position_base_m=[.36536324,.04280226,.02307460],orientation_wxyz=quat.tolist())
    generated=plan_cleanup(config,kinds=("gamepad","speaker"),object_poses_base={"gamepad":pose})
    assert config==original
    g=generated["grasps"]["gamepad"]
    assert g["pose_source"]=="measured_settled"
    assert generated["grasps"]["speaker"]["pose_source"]=="nominal_config"
    kin=PiperKinematics(ROOT/config["scene"]["robot_urdf"],tip_offset=.1358)
    descent=[w for w in generated["waypoints"] if w["phase"]=="gamepad:descend"]
    normal=orientation.as_matrix()[:,2]
    for w in descent:
        transform=kin.fk(np.asarray(w["action"][:6]))
        np.testing.assert_allclose(transform[:3,:3],g["grasp_rotation_base"],atol=1e-4)
        delta=np.asarray(w["tcp_base_m"])-g["grasp_tcp_base_m"]
        np.testing.assert_allclose(np.cross(delta,normal),0.,atol=5e-4)
    # Carrying a tilted source must still orient the object upright in the box.
    relative=np.asarray(g["grasp_rotation_base"]).T@orientation.as_matrix()
    placed_rotation=np.asarray(g["placement_rotation_base"])@relative
    np.testing.assert_allclose(placed_rotation,Rotation.from_euler("z",config["bin"]["yaw_deg"],degrees=True).as_matrix(),atol=1e-10)
    _assert_direct_second_pickup(generated)


@pytest.mark.parametrize("poses",[
    {"unknown":{}}, {"gamepad":{"position_world":[0,0,0],"orientation_wxyz":[1,0,0,0]}},
    {"gamepad":{"position_base_m":[.35,0,float("nan")],"orientation_wxyz":[1,0,0,0]}},
    {"gamepad":{"position_base_m":[.35,0,.03],"orientation_wxyz":[0,0,0,0]}},
    {"gamepad":{"position_base_m":[.35,0,.03],"orientation_wxyz":[0,1,0,0]}},
])
def test_invalid_measured_object_pose_is_not_silently_ignored(config,poses):
    with pytest.raises(ValueError):
        plan_cleanup(config,kinds=("gamepad","speaker"),object_poses_base=poses)


def test_observed_pose_conversion_is_invariant_to_parallel_environment_translation():
    quat=Rotation.from_euler("xyz",[-9.6,0,31.],degrees=True).as_quat()[[3,0,1,2]]
    world=np.array([.365,.043,.789])
    base=np.array([0,0,.766])
    offset=np.array([6.,-6.,0.])
    def snapshot(position):
        return {"objects":{"gamepad":dict(position_world=position.tolist(),
                                          orientation_wxyz=quat.tolist(),active=True)}}
    first=smoke.measured_object_poses(snapshot(world),base,["gamepad"])
    peer=smoke.measured_object_poses(snapshot(world+offset),base+offset,["gamepad"])
    np.testing.assert_allclose(first["gamepad"]["position_base_m"],peer["gamepad"]["position_base_m"],atol=1e-12)
    np.testing.assert_allclose(first["gamepad"]["orientation_wxyz"],quat)


def test_parallel_completion_only_pads_final_hold_without_replaying_actions(plan):
    short,long=deepcopy(plan),deepcopy(plan)
    long["actions"].extend([deepcopy(long["actions"][-1])]*7)
    long["phases"].extend(["settled"]*7)
    original=np.array(short["actions"])
    smoke.equalize_plan_lengths({0:short,1:long})
    np.testing.assert_array_equal(short["actions"][:len(original)],original)
    np.testing.assert_array_equal(short["actions"][len(original):],np.tile(original[-1],(7,1)))
    assert short["phases"][-7:]==["settled"]*7
    assert short["parallel_padding_frames"]==7 and long["parallel_padding_frames"]==0
    assert short["duration_s"]==long["duration_s"]


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
    assert config["scene_config"]=="configs/calibration/playground_20261113_appearance.json"
    assert generated["kinds"]==["gamepad","speaker"]
    _assert_direct_second_pickup(generated)
    canonical=json.loads((ROOT/"configs/rl_nodes/tabletop_cleanup_real_layout.json").read_text())["backend"]["config"]
    assert generated["bin"]==canonical["bin"]
    from pressb.robots.piper.kinematics import PiperKinematics
    scene=json.loads((ROOT/canonical["scene_config"]).read_text())
    kin=PiperKinematics(ROOT/scene["robot_urdf"],tip_offset=.1358)
    state=np.asarray(generated["expected_initial_state"])
    np.testing.assert_allclose(kin.fk(state[:6])[:3,3],[.36,-.03,.16-scene["robot_base_height_above_table_m"]],atol=1e-5)
    assert state[6]==pytest.approx(.1)
    assert report["status"]=="planned" and report["physical_execution"] is False
