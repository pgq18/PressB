#!/usr/bin/env python3
"""Record two physical pick/place operations in the shared Isaac cleanup task.

Simulation only: only seven-dimensional joint/gripper targets go through step.
The props are not attached, teleported, or manually moved after reset. The full
script is a diagnostic demonstrator, not a learned-policy success evaluation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--gpu",type=int,default=1)
    p.add_argument("--num-envs",type=int,default=1)
    p.add_argument("--config",type=Path,default=ROOT/"configs/rl_nodes/tabletop_cleanup_real_layout.json")
    p.add_argument("--objects",nargs=2,choices=("gamepad","speaker","lidar"),
                   help="Override config.randomization.fixed_subset; otherwise use its fixed pair")
    p.add_argument("--bin-xy",type=float,nargs=2)
    p.add_argument("--grasp-opening",action="append",nargs=2,metavar=("KIND","METRES"),default=[],
                   help="Override one object's approach/release opening for this run; closing target stays unchanged")
    p.add_argument("--max-joint-speed",type=float,default=20.,help="Degrees per second; simulation only")
    p.add_argument("--settle-seconds",type=float,default=1.,
                   help="Hold the initial robot pose through normal steps after reset before planning")
    p.add_argument("--settled-grasps",action=argparse.BooleanOptionalAction,default=True,
                   help="Plan from each environment's measured post-settling object poses")
    p.add_argument("--video",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--plan-only",action="store_true")
    return p


def measured_object_poses(snapshot, base_world, kinds):
    """Convert diagnostic rigid-body origins into one robot's base frame."""
    import numpy as np
    base = np.asarray(base_world,dtype=float)
    if base.shape!=(3,) or not np.isfinite(base).all():
        raise ValueError("Robot base must be a finite world position")
    result={}
    for kind in kinds:
        obj=snapshot["objects"][kind]
        position=np.asarray(obj["position_world"],dtype=float)
        quaternion=np.asarray(obj["orientation_wxyz"],dtype=float)
        if (obj.get("active") is not True or position.shape!=(3,) or quaternion.shape!=(4,)
                or not np.isfinite(position).all() or not np.isfinite(quaternion).all()
                or np.linalg.norm(quaternion)<1e-12):
            raise ValueError(f"Cannot plan from an inactive or invalid {kind} observation")
        result[kind]=dict(position_base_m=(position-base).tolist(),
                          orientation_wxyz=(quaternion/np.linalg.norm(quaternion)).tolist())
    return result


def equalize_plan_lengths(plans):
    """Hold completed environments at home while a peer finishes its own plan."""
    longest=max(len(plan["actions"]) for plan in plans.values())
    for plan in plans.values():
        count=len(plan["actions"])
        padding=longest-count
        plan["unpadded_action_count"]=count
        plan["parallel_padding_frames"]=padding
        if padding:
            plan["actions"].extend([deepcopy(plan["actions"][-1]) for _ in range(padding)])
            plan["phases"].extend(["settled"]*padding)
            waypoint=deepcopy(plan["waypoints"][-1])
            waypoint.update(phase="settled",start_frame=count,arrival_frame=count,
                            end_frame=longest-1,hold_seconds=padding/plan["fps"],minimum_duration_s=0.)
            plan["waypoints"].append(waypoint)
        plan["duration_s"]=longest/plan["fps"]
    return plans


def assess_execution(history, plan):
    """Require complete playback, measured lift, release and stable success."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    if not history:
        return dict(passed=False,reason="no physical samples")
    final = history[-1]
    physical = final["info"].get("physical_outcome",{})
    window = history[-max(2,round(plan["fps"]*.5)):]
    outcomes = {}
    for kind in plan["kinds"]:
        positions = np.asarray([row["objects"][kind]["position_world"] for row in history])
        stable = np.asarray([row["objects"][kind]["position_world"] for row in window])
        rotations = Rotation.from_quat(np.asarray([row["objects"][kind]["orientation_wxyz"] for row in window])[:,[1,2,3,0]])
        state = physical.get("objects",{}).get(kind,{})
        measured = dict(max_lift_m=float(np.max(positions[:,2])-positions[0,2]),
                        stable_position_excursion_m=float(np.max(np.linalg.norm(stable-stable[0],axis=1))),
                        stable_rotation_excursion_rad=float(np.max((rotations[0].inv()*rotations).magnitude())),
                        physical_outcome=deepcopy(state))
        measured["passed"] = bool(measured["max_lift_m"]>.025 and
            measured["stable_position_excursion_m"]<.002 and measured["stable_rotation_excursion_rad"]<math.radians(4) and
            all(state.get(key) is True for key in ("active","grasped_and_lifted","contained","supported","released","low_velocity")))
        outcomes[kind] = measured
    state = np.asarray(final["state"])
    full = final["phase"]=="settled" and final["action_index"]==len(plan["actions"])-1
    home_error = float(np.max(np.abs(state[:6]-plan["expected_initial_state"][:6])))
    result = dict(full_plan_executed=full,task_success=final["info"].get("success") is True,
                  physical_success=physical.get("success") is True,objects=outcomes,
                  final_gripper_width_m=float(state[6]),max_home_joint_error_rad=home_error)
    result["passed"] = bool(full and result["task_success"] and result["physical_success"] and
                            all(row["passed"] for row in outcomes.values()) and state[6]>.040 and home_error<.030)
    return result


def run(args):
    import numpy as np
    from pressb.tasks.tabletop_cleanup.parallel import TabletopCleanupTask,create_backend
    from pressb.tasks.tabletop_cleanup.planning import plan_cleanup
    from pressb.runtime.contract import decode_image
    from pressb.runtime.recording import Video,write
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True,exist_ok=False)
    source = json.loads(args.config.read_text())
    config = deepcopy(source.get("backend",{}).get("config",source))
    config.update(num_envs=args.num_envs,gpu=args.gpu,horizon=1,terminate_on_success=False,
                  max_episode_seconds=max(120.,float(config.get("max_episode_seconds",120.))),
                  settling_ticks=120,smoothing={"mode":"none"},
                  render_resolution=[640,480],image_output_resolution=[640,480],
                  observation_render_frames=1,reset_render_frames=2,log_dir=str(output/"simulation"))
    selected=args.objects or config.get("randomization",{}).get("fixed_subset") or ["speaker","lidar"]
    config.setdefault("randomization",{}).update(enabled=False,fixed_subset=selected)
    if args.bin_xy is not None:
        config.setdefault("bin",{})["center_xy"] = args.bin_xy
    report = dict(kind="tabletop_cleanup_shared_parallel_physics",status="running",simulation_only=True,
                  hardware_access=False,attachments_or_object_teleportation=False,num_envs=args.num_envs,
                  learned_policy_evaluation=False,started_unix=time.time(),closed=False)
    backend,pipes,history = None,{},{}
    try:
        normalized = TabletopCleanupTask(config).config
        openings={kind:float(width) for kind,width in args.grasp_opening}
        plan = plan_cleanup(normalized,kinds=selected,max_joint_speed_deg_s=args.max_joint_speed,
                            grasp_openings_m=openings)
        config["max_episode_seconds"] = max(config["max_episode_seconds"],
            float(math.ceil(plan["duration_s"]+args.settle_seconds+1.)))
        write(output/"plan.json",plan);write(output/"config.json",config)
        report["plan_duration_s"] = plan["duration_s"]
        if args.plan_only:
            report.update(status="planned",physical_execution=False,
                          planning_pose_source="nominal_config_no_physics_measurement",
                          settled_grasps_requested=args.settled_grasps)
            return 0
        backend = create_backend(config)
        from pressb.runtime.parallel import SharedWorldBackend
        if not isinstance(backend,SharedWorldBackend):
            raise RuntimeError("Cleanup requires the common SharedWorldBackend")
        report["metadata"] = backend.describe()
        ids = list(range(args.num_envs))
        reset_objects = {kind:{key:plan["grasps"][kind]["object"].get(key,0.) for key in ("center_xy","yaw_deg")}
                         for kind in plan["kinds"]}
        rows = backend.reset(ids,20261109,{"objects":reset_objects})
        reset_snapshot = backend.snapshot()
        report["reset_snapshot"]=reset_snapshot
        settle_frames=math.ceil(args.settle_seconds*plan["fps"])
        home=normalized["scene"]["initial_state_joint_rad_m"]
        with (output/"settling.jsonl").open("w") as stream:
            for index in range(settle_frames):
                if not all(backend.ready[eid] for eid in ids):
                    raise RuntimeError("An environment ended before object settling completed")
                rows=backend.step([dict(env_id=eid,actions=[home]) for eid in ids])
                snapshots=backend.snapshot(ids)
                for row in rows:
                    eid=row["env_id"]
                    record=dict(env_id=eid,settling_action_index=index,phase="initial_settling",
                        elapsed_s=float(backend.ticks[eid])*backend.execution["physics_dt"],
                        command=home,state=row["observation"]["state"],
                        objects=snapshots[str(eid)]["objects"],info=deepcopy(row["info"]),
                        terminated=row["terminated"],truncated=row["truncated"])
                    stream.write(json.dumps(record,allow_nan=False)+"\n")
        first = backend.snapshot()
        if not all(backend.ready[eid] for eid in ids):
            raise RuntimeError("An environment ended during object settling")
        report.update(settled_snapshot=first,
            settling=dict(requested_seconds=args.settle_seconds,control_steps=settle_frames,
                control_hz=plan["fps"],actual_seconds=settle_frames/plan["fps"],
                trace="settling.jsonl",included_in_video=False,
                velocities_by_env={str(eid):{kind:first[str(eid)]["objects"][kind]["velocity"]
                                            for kind in selected} for eid in ids}),
            planning_pose_source="measured_post_settling" if args.settled_grasps else "nominal_config")
        plans={}
        poses_by_env={}
        for eid in ids:
            base_world=(np.asarray(backend.task.scene["robot_base_position_world"])
                        +np.asarray(backend.engine.offsets[eid]))
            poses_by_env[str(eid)]=measured_object_poses(first[str(eid)],base_world,selected)
            kwargs=dict(object_poses_base=poses_by_env[str(eid)]) if args.settled_grasps else {}
            plans[eid]=plan_cleanup(normalized,kinds=selected,max_joint_speed_deg_s=args.max_joint_speed,
                                    grasp_openings_m=openings,**kwargs)
        equalize_plan_lengths(plans)
        plan=plans[ids[0]]
        required_seconds=plan["duration_s"]+settle_frames/plan["fps"]
        if required_seconds>=config["max_episode_seconds"]:
            raise ValueError(f"Measured plan and settling need {required_seconds:.3f} seconds, "
                             f"exceeding the configured {config['max_episode_seconds']:.3f} second episode")
        report.update(plan_duration_s=plan["duration_s"],object_poses_base_by_env=poses_by_env,
                      unpadded_action_counts={str(eid):plans[eid]["unpadded_action_count"] for eid in ids})
        write(output/"plan.json",plan)
        write(output/"plans_by_env.json",{str(eid):plans[eid] for eid in ids})
        write(output/"settled_snapshot.json",first)
        rgb_cache,history = {},{i:[] for i in ids}
        def frames(rows,label=None):
            for row in rows:
                eid = row["env_id"]
                rgb_cache[eid] = {name:np.asarray(decode_image(value)) for name,value in row["observation"]["images"].items()}
                if set(rgb_cache[eid])!={"up","wrist"}:
                    raise RuntimeError("Recording requires up and wrist cameras")
                if label:
                    from PIL import Image
                    for name,rgb in rgb_cache[eid].items():
                        Image.fromarray(rgb).save(output/f"{label}_env{eid}_{name}.png")
            if args.video:
                columns=[]
                for eid in ids:
                    column=np.concatenate([rgb_cache[eid]["up"],rgb_cache[eid]["wrist"]],axis=0)
                    columns.append(column)
                    if eid not in pipes:
                        pipes[eid]=Video(output/f"env{eid}_up_wrist.mp4",(column.shape[1],column.shape[0]),plan["fps"])
                    pipes[eid].add(np.ascontiguousarray(column))
                if len(ids)>1:
                    grid=np.concatenate(columns,axis=1)
                    if "parallel" not in pipes:
                        pipes["parallel"]=Video(output/"parallel_up_wrist.mp4",(grid.shape[1],grid.shape[0]),plan["fps"])
                    pipes["parallel"].add(np.ascontiguousarray(grid))
        frames(rows,"initial")
        with (output/"physics.jsonl").open("w") as stream:
            previous_phases={}
            for index in range(len(plan["actions"])):
                live=[eid for eid in ids if backend.ready[eid]]
                if not live:
                    break
                rows=backend.step([dict(env_id=eid,actions=[plans[eid]["actions"][index]]) for eid in live])
                snapshots=backend.snapshot(live)
                for row in rows:
                    eid=row["env_id"]
                    command,phase=plans[eid]["actions"][index],plans[eid]["phases"][index]
                    record=dict(env_id=eid,action_index=index,phase=phase,
                        elapsed_s=float(backend.ticks[eid])*backend.execution["physics_dt"],command=command,
                        playback_elapsed_s=(index+1)/plan["fps"],
                        state=row["observation"]["state"],objects=snapshots[str(eid)]["objects"],
                        info=deepcopy(row["info"]),terminated=row["terminated"],truncated=row["truncated"])
                    history[eid].append(record);stream.write(json.dumps(record,allow_nan=False)+"\n")
                    if phase!=previous_phases.get(eid):
                        print(json.dumps(dict(env_id=eid,frame=index,total=len(plans[eid]["actions"]),phase=phase)),flush=True)
                        previous_phases[eid]=phase
                frames(rows,"final" if index==len(plan["actions"])-1 else None)
        audits={str(eid):assess_execution(history[eid],plans[eid]) for eid in ids}
        report.update(status="passed" if all(item["passed"] for item in audits.values()) else "failed",
            physical_execution=True,episodes=audits,initial_snapshot=first,final_snapshot=backend.snapshot(),
            timings=deepcopy(backend.timings),video_paths={str(key):str(pipe.path) for key,pipe in pipes.items()},
            video_fps=plan["fps"],video_frame_counts={str(key):pipe.count for key,pipe in pipes.items()})
        active=[eid for eid in ids if backend.ready[eid]]
        if active:
            backend.end_episode(active)
    except Exception as exc:
        import traceback
        report.update(status="failed",error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
    finally:
        for pipe in pipes.values():pipe.close()
        report["finished_unix"]=time.time();write(output/"report.json",report)
        if backend is not None:
            backend.close()
            report["closed"]=True;write(output/"report.json",report)
    return 0 if report["status"] in ("planned","passed") else 1


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    if args.num_envs<1 or args.gpu<0 or (args.objects is not None and len(set(args.objects))!=2):
        p.error("Use a positive num-envs, nonnegative GPU and two distinct objects")
    if (not math.isfinite(args.settle_seconds) or args.settle_seconds<0
            or (args.settled_grasps and args.settle_seconds<=0)):
        p.error("--settle-seconds must be finite and positive when using measured settled grasps")
    try:
        openings={kind:float(width) for kind,width in args.grasp_opening}
    except ValueError:
        p.error("--grasp-opening needs an object kind and numeric width in metres")
    if (len(openings)!=len(args.grasp_opening) or set(openings)-set(args.objects or ("gamepad","speaker","lidar")) or
            any(not math.isfinite(width) or width<=0 for width in openings.values())):
        p.error("Opening overrides require distinct selected kinds and positive finite widths")
    if args.output.expanduser().exists():p.error("--output must be new")
    return run(args)


if __name__=="__main__":
    raise SystemExit(main())
