#!/usr/bin/env python3
"""Validate the real Isaac playground backend; never accesses real hardware."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run(args):
    import numpy as np
    from pressb.runtime.contract import decode_image
    from pressb.runtime.simulation import SimulationService
    from pressb.tasks.playground.backend import PlaygroundBackend
    from pressb.robots.piper.kinematics import PiperKinematics

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = {"scene_config": str(args.scene.resolve()), "gpu": args.gpu, "headless": True,
              "num_envs": 1, "horizon": 8, "max_episode_seconds": .8,
              "smoothing": {"mode": args.smoothing, "alpha": .25}}
    report = {"status": "running", "closed": False, "simulation_only": True,
              "native_shutdown_exits_process": True, "child_pid": os.getpid(),
              "config": config, "checks": {}, "observations": [], "steps": []}
    backend = service = None
    started = time.monotonic()

    def save():
        report["wall_seconds"] = time.monotonic()-started
        (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    def require(value, message):
        if not value:
            raise AssertionError(message)

    def observe(row, name):
        from pxr import UsdGeom
        runtime = backend.environment.runtime
        state = np.asarray(row["observation"]["state"])
        require(state.shape == (7,) and np.isfinite(state).all(), "Invalid measured state")
        images = {}
        for camera, wire in row["observation"]["images"].items():
            image = decode_image(wire)
            pixels = np.asarray(image)
            require(image.size == (640, 480) and float(pixels.std()) > 5., "Empty or wrong-sized image")
            image.save(output/f"{name}_{camera}.png")
            images[camera] = {"size": list(image.size), "std": float(pixels.std()),
                              "sha256": hashlib.sha256(pixels.tobytes()).hexdigest()}
            spec = backend.config["scene"][camera+"_camera"]
            prim = UsdGeom.Camera(runtime.stage.GetPrimAtPath(runtime.camera_specs[camera][0]))
            focal = prim.GetFocalLengthAttr().Get()
            horizontal = prim.GetHorizontalApertureAttr().Get()
            vertical = prim.GetVerticalApertureAttr().Get()
            width, height = spec["resolution"]
            measured_k = [focal*width/horizontal, focal*height/vertical,
                width/2-prim.GetHorizontalApertureOffsetAttr().Get()*width/horizontal,
                height/2+prim.GetVerticalApertureOffsetAttr().Get()*height/vertical]
            expected_k = [spec["fx"], spec["fy"], spec.get("cx", width/2), spec.get("cy", height/2)]
            require(np.max(np.abs(np.subtract(measured_k, expected_k))) < .001,
                    "Rendered intrinsics differ from configured calibration")
            images[camera]["usd_intrinsics_fx_fy_cx_cy"] = measured_k
        require(set(images) == {"up", "wrist"}, "Missing camera")
        kin = PiperKinematics(backend.config["scene"]["robot_urdf"], tip_offset=0.)
        fk = kin.link_transforms(state[:6])
        cache = UsdGeom.XformCache()
        error = 0.
        for link, expected in fk.items():
            path = "/World/Piper/"+link
            prim = runtime.stage.GetPrimAtPath(path)
            if not prim or link == "base_link":
                continue
            expected = expected.copy()
            expected[:3, 3] += runtime.base
            actual = np.asarray(cache.GetLocalToWorldTransform(prim)).T
            error = max(error, float(np.max(np.abs(expected-actual))))
        wrist_spec = backend.config["scene"]["wrist_camera"]
        expected = fk["link6"] @ np.asarray(wrist_spec["link6_from_camera_usd"])
        expected[:3, 3] += runtime.base
        actual = np.asarray(cache.GetLocalToWorldTransform(
            runtime.stage.GetPrimAtPath(runtime.wrist_info["camera_path"]))).T
        wrist_error = float(np.max(np.abs(expected-actual)))
        require(error < 2e-4 and wrist_error < 2e-4, "Rendered robot/wrist transforms disagree with measured FK")
        report["observations"].append({"name": name, "state": state.tolist(), "images": images,
            "link_transform_max_error": error, "wrist_transform_max_error": wrist_error})
        return state

    try:
        backend = PlaygroundBackend(config)
        service = SimulationService(backend)
        metadata = service.health()
        report["metadata"] = metadata
        require(metadata["num_envs"] == 1 and not metadata["execution"]["vectorized"], "Wrong environment count")
        reset_request = {"run_id": "smoke", "request_id": "reset-0", "env_ids": [0], "seed": 7, "options": {}}
        initial_response = service.reset(reset_request)
        row = initial_response["items"][0]
        initial = observe(row, "initial")
        before_time = backend.environment.runtime.world.current_time
        require(service.reset(reset_request) == initial_response, "Reset retry changed response")
        require(backend.environment.runtime.world.current_time == before_time, "Reset retry advanced physics")
        target = np.clip(initial, metadata["action"]["lower"], metadata["action"]["upper"])
        target[0] += .06
        target[6] = .025
        for index in range(3):
            request = {"run_id": "smoke", "request_id": f"step-{index}", "items": [{
                "env_id": 0, "episode_id": row["episode_id"], "step_id": row["step_id"],
                "actions": np.tile(target, (8, 1)).tolist()}]}
            response = service.step(request)
            row = response["items"][0]
            state = observe(row, f"step_{index}")
            after_time = backend.environment.runtime.world.current_time
            require(service.step(request) == response, "Step retry changed response")
            require(backend.environment.runtime.world.current_time == after_time, "Step retry advanced physics")
            require(row["info"]["physics_steps"] == 32 and row["info"]["elapsed_control_steps"] == 8,
                    "Incorrect chunk duration")
            require(math.isclose(row["discount"], .99**8, rel_tol=1e-6), "Incorrect chunk discount")
            require(row["reward"] == 0 and not row["terminated"] and not row["info"]["success"], "Invented task outcome")
            require(row["truncated"] == (index == 2), "Incorrect time limit")
            report["steps"].append({key: row[key] for key in ("reward", "discount", "terminated", "truncated", "info")})
            save()
        require(abs(state[0]-initial[0]) > .02, "Measured J1 did not move")
        require(abs(state[6]-.025) < .002, "Gripper failed to follow SI width target")
        require(np.max(np.abs(state[:6]-initial[:6])) < .1, "Unexpected large simulated movement")
        report["checks"]["measured_j1_change_rad"] = float(state[0]-initial[0])
        report["checks"]["gripper_width_m"] = float(state[6])
        if args.smoothing == "joint_ema":
            first = np.asarray(report["steps"][0]["info"]["applied_actions_rad_m"][0])
            require(np.allclose(first[:6], .25*target[:6]+.75*initial[:6], atol=1e-7), "EMA not applied")
            require(math.isclose(first[6], target[6], abs_tol=1e-7), "Gripper was smoothed")
        reset_request.update(request_id="reset-1", seed=8)
        second = service.reset(reset_request)["items"][0]
        reset_state = observe(second, "reset_again")
        require(second["episode_id"] != row["episode_id"] and second["step_id"] == 0, "Reset identity not renewed")
        require(np.max(np.abs(reset_state-initial)) < .002, "Reset failed to restore observed start")
        retired_time = backend.environment.runtime.world.current_time
        service.end_episode({"run_id": "smoke", "request_id": "retire", "items": [{
            "env_id": 0, "episode_id": second["episode_id"], "step_id": second["step_id"]}]})
        require(backend.environment.runtime.world.current_time == retired_time, "Retirement advanced physics")
        report["checks"].update(retry_idempotent=True, timed_truncation=True, reset_after_terminal=True,
                                measured_state_changed=True, images_valid=True, fk_camera_agrees=True)
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    finally:
        report["operations_status"] = report["status"]
        if service is not None:
            report["close_attempted"] = True
            # Fast native shutdown deliberately exits the process from close().
            # Record the boundary after camera/world cleanup, then let the
            # independent supervisor verify actual child exit and exit code.
            environment = backend.environment
            if environment is not None and environment.app is not None:
                native_close = environment.app.close

                def recorded_native_close(*close_args, **close_kwargs):
                    report["component_cleanup_completed"] = environment.runtime is None
                    report["native_close_started"] = True
                    save()
                    return native_close(*close_args, **close_kwargs)

                environment.app.close = recorded_native_close
            save()
            try:
                service.close()
                report["closed"] = True
                report["close_returned"] = True
            except BaseException as error:
                report["status"] = "failed"
                report["close_error"] = f"{type(error).__name__}: {error}"
        save()
        print(json.dumps({key: report.get(key) for key in ("status", "error", "closed", "wall_seconds")}), flush=True)
    return 0 if report["status"] == "passed" and report["closed"] else 1


def supervise(args):
    """Isaac fast shutdown skips Python finally blocks; verify it externally."""
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.with_name(output.name+".log")
    command = [sys.executable, str(Path(__file__).resolve()), "--child", "--scene", str(args.scene.resolve()),
               "--gpu", str(args.gpu), "--smoothing", args.smoothing, "--output", str(output)]
    started = time.monotonic()
    with log.open("x") as stream:
        try:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
            exit_code, error = process.returncode, None
        except subprocess.TimeoutExpired:
            exit_code, error = None, "Isaac child exceeded 180 seconds and was terminated"
    report_path = output/"report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {"status": "failed", "operations_status": "missing"}
    report.update(child_exit_code=exit_code, process_exited=True, supervisor_seconds=time.monotonic()-started,
                  process_log=str(log), close_returned=report.get("close_returned", False))
    verified = (exit_code == 0 and report.get("operations_status") == "passed"
                and report.get("component_cleanup_completed") is True and report.get("native_close_started") is True)
    report["native_shutdown_verified"] = verified
    report["closed"] = verified
    report["status"] = "passed" if verified else "failed"
    if error:
        report["supervisor_error"] = error
    elif not verified:
        report["supervisor_error"] = "Operations or cleanup/native shutdown verification failed"
    output.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: report.get(key) for key in ("status", "error", "closed", "close_returned",
        "child_exit_code", "native_shutdown_verified", "supervisor_seconds")}), flush=True)
    return 0 if verified else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=ROOT/"configs/piper_playground.json")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--smoothing", choices=("none", "joint_ema"), default="none")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    raise SystemExit(run(args) if args.child else supervise(args))
