#!/usr/bin/env python3
"""Physically grasp red, lift, stack on blue, open and return in shared Isaac.

Simulation only. The task plugin authors the objects; this script only sends
joint targets through reset/step. No real robot, attachment or object teleport
is used during execution. Both cameras are recorded at the control frequency.
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
    p.add_argument("--num-envs",type=int,default=2)
    p.add_argument("--scene-config",default="configs/calibration/playground_20261113_appearance.json")
    p.add_argument("--red-xy",type=float,nargs=2,help="Override the plugin's red cube centre")
    p.add_argument("--blue-xy",type=float,nargs=2,help="Override the plugin's blue cube centre")
    p.add_argument("--red-edge",type=float,help="Override the plugin's red cube size")
    p.add_argument("--blue-edge",type=float,help="Override the plugin's blue cube size")
    p.add_argument("--grasp-height-above-table",type=float,default=.012)
    p.add_argument("--lift-height-above-table",type=float,default=.090)
    p.add_argument("--close-width",type=float,default=.026)
    p.add_argument("--max-joint-speed",type=float,default=20.,help="Degrees per second, simulation plan only")
    p.add_argument("--video",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--plan-only",action="store_true")
    return p


def assess_execution(history, plan):
    """Independent end-state and stable-window audit, beside the task success."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    if not history:
        return {"passed":False,"reason":"no physical samples"}
    final = history[-1]
    edge = min(plan["source"]["size"][:2])
    source_height,target_height = plan["source"]["size"][2],plan["target"]["size"][2]
    positions = np.asarray([x["objects"]["red"]["position_world"] for x in history])
    blue = np.asarray(final["objects"]["blue"]["position_world"])
    red = positions[-1]
    window = history[-max(2,round(plan["fps"]*.5)):]
    stable = np.asarray([x["objects"]["red"]["position_world"] for x in window])
    rotations = Rotation.from_quat(np.asarray([x["objects"]["red"]["orientation_wxyz"] for x in window])[:,[1,2,3,0]])
    excursion = float(np.max(np.linalg.norm(stable-stable[0],axis=1)))
    rotation_excursion = float(np.max((rotations[0].inv()*rotations).magnitude()))
    axis = rotations[-1].as_matrix()[:,2]
    final_state = np.asarray(final["state"])
    joint_error = float(np.max(np.abs(final_state[:6]-plan["expected_initial_state"][:6])))
    max_lift = float(positions[:,2].max()-positions[0,2])
    xy_error = float(np.linalg.norm(red[:2]-blue[:2]))
    separation_error = float(abs(red[2]-blue[2]-(source_height+target_height)/2))
    full = final["phase"]=="settled" and final["action_index"]==len(plan["actions"])-1
    task_success = final["info"].get("success") is True
    result = dict(full_plan_executed=full,task_success=task_success,red_blue_xy_error_m=xy_error,
        center_height_separation_error_m=separation_error,red_final_upright_cosine=float(axis[2]),
        stable_window_samples=len(window),red_position_excursion_m=excursion,
        red_rotation_excursion_rad=rotation_excursion,max_red_lift_m=max_lift,
        final_gripper_width_m=float(final_state[6]),max_home_joint_error_rad=joint_error)
    result["passed"] = bool(full and task_success and max_lift>target_height+.010 and xy_error<edge*.25
        and separation_error<.004 and axis[2]>math.cos(math.radians(12))
        and excursion<.001 and rotation_excursion<math.radians(3)
        and final_state[6]>.050 and joint_error<.030)
    return result


def run(args):
    import numpy as np
    from pressb.tasks.cube_stack.parallel import CubeStackTask,create_backend
    from pressb.tasks.cube_stack.planning import plan_stack
    from pressb.runtime.contract import decode_image
    from pressb.runtime.recording import Video,write
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True,exist_ok=False)
    config = dict(scene_config=args.scene_config,num_envs=args.num_envs,gpu=args.gpu,horizon=1,
        max_episode_seconds=60.,terminate_on_success=False,settling_ticks=120,
        randomization={"enabled":False},
        smoothing={"mode":"none"},render_resolution=[640,480],image_output_resolution=[640,480],
        observation_render_frames=1,reset_render_frames=2,log_dir=str(output/"simulation"))
    cubes = {}
    for color in ("red","blue"):
        changes = {}
        if getattr(args,color+"_xy") is not None:
            changes["center_xy"] = getattr(args,color+"_xy")
        if getattr(args,color+"_edge") is not None:
            changes["edge_m"] = getattr(args,color+"_edge")
        if changes:
            cubes[color] = changes
    if cubes:
        config["cubes"] = cubes
    report = dict(kind="cube_stack_shared_parallel_physics",status="running",simulation_only=True,
        hardware_access=False,attachments_or_object_teleportation=False,num_envs=args.num_envs,
        learned_policy_evaluation=False,started_unix=time.time(),closed=False)
    backend,pipes,history = None,{},{}
    try:
        # Pure CPU plugin construction and complete IK planning precede Kit.
        normalized = CubeStackTask(config).config
        plan = plan_stack(normalized["scene"],grasp_height_above_table=args.grasp_height_above_table,
            lift_height_above_table=args.lift_height_above_table,close_width=args.close_width,
            max_joint_speed_deg_s=args.max_joint_speed)
        config["max_episode_seconds"] = float(math.ceil(plan["duration_s"]+1.))
        write(output/"plan.json",plan);write(output/"config.json",config)
        report["plan_duration_s"] = plan["duration_s"]
        if args.plan_only:
            report.update(status="planned",physical_execution=False)
            return 0
        backend = create_backend(config)
        from pressb.runtime.parallel import SharedWorldBackend
        if not isinstance(backend,SharedWorldBackend):
            raise RuntimeError("cube_stack must use the common SharedWorldBackend")
        report["metadata"] = backend.describe()
        ids = list(range(args.num_envs))
        reset_objects = {color:{"center_xy":plan[key]["center_xy"],"yaw_deg":plan[key].get("yaw_deg",0.)}
                         for color,key in (("red","source"),("blue","target"))}
        rows = backend.reset(ids,20261109,{"objects":reset_objects})
        first = backend.snapshot()
        rgb_cache = {}
        history = {i:[] for i in ids}
        def frames(rows, label=None):
            for row in rows:
                eid = row["env_id"]
                rgb_cache[eid] = {name:np.asarray(decode_image(image)) for name,image in row["observation"]["images"].items()}
                if set(rgb_cache[eid])!={"up","wrist"}:
                    raise RuntimeError("Stack recording requires up and wrist cameras")
                if label:
                    from PIL import Image
                    for camera,rgb in rgb_cache[eid].items():
                        Image.fromarray(rgb).save(output/f"{label}_env{eid}_{camera}.png")
            if args.video:
                columns = []
                for eid in ids:
                    column = np.concatenate([rgb_cache[eid]["up"],rgb_cache[eid]["wrist"]],axis=0)
                    columns.append(column)
                    if eid not in pipes:
                        pipes[eid] = Video(output/f"env{eid}_up_wrist.mp4",(column.shape[1],column.shape[0]),plan["fps"])
                    pipes[eid].add(np.ascontiguousarray(column))
                grid = np.concatenate(columns,axis=1)
                if "parallel" not in pipes:
                    pipes["parallel"] = Video(output/"parallel_up_wrist.mp4",(grid.shape[1],grid.shape[0]),plan["fps"])
                pipes["parallel"].add(np.ascontiguousarray(grid))
        frames(rows,"initial")
        with (output/"physics.jsonl").open("w") as stream:
            for index,(command,phase) in enumerate(zip(plan["actions"],plan["phases"])):
                live = [i for i in ids if backend.ready[i]]
                if not live:
                    break
                rows = backend.step([dict(env_id=i,actions=[command]) for i in live])
                snapshots = backend.snapshot(live)
                for row in rows:
                    eid = row["env_id"]
                    record = dict(env_id=eid,action_index=index,phase=phase,
                        elapsed_s=float(backend.ticks[eid])*backend.execution["physics_dt"],
                        command=command,state=row["observation"]["state"],
                        objects=snapshots[str(eid)]["objects"],info=deepcopy(row["info"]),
                        terminated=row["terminated"],truncated=row["truncated"])
                    history[eid].append(record)
                    stream.write(json.dumps(record,allow_nan=False)+"\n")
                frames(rows,"final" if index==len(plan["actions"])-1 else None)
        audits = {str(i):assess_execution(history[i],plan) for i in ids}
        report.update(status="passed" if all(item["passed"] for item in audits.values()) else "failed",
            physical_execution=True,episodes=audits,initial_snapshot=first,final_snapshot=backend.snapshot(),
            timings=deepcopy(backend.timings),video_paths={str(key):str(pipe.path) for key,pipe in pipes.items()},
            video_fps=plan["fps"],video_frame_counts={str(key):pipe.count for key,pipe in pipes.items()})
        active = [i for i in ids if backend.ready[i]]
        if active:
            backend.end_episode(active)
    except Exception as exc:
        import traceback
        report.update(status="failed",error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
    finally:
        for pipe in pipes.values():
            pipe.close()
        report["finished_unix"] = time.time()
        write(output/"report.json",report)
        if backend is not None:
            backend.close()
            report["closed"] = True
            write(output/"report.json",report)
    return 0 if report["status"] in ("planned","passed") else 1


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.num_envs<1 or args.gpu<0:
        p.error("Use positive num-envs and a nonnegative GPU")
    if args.output.expanduser().exists():
        p.error("--output must be new")
    return run(args)


if __name__=="__main__":
    raise SystemExit(main())
