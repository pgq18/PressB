"""Serve one verified PiPER checkpoint over a loopback-only HTTP connection.

Run with ``python -m scripts.serve_piper_policy --checkpoint ...``. Forward
the port through the existing SSH connection for an Isaac Sim client. Each
request carries current RGB observations and a measured base_link TCP pose;
the server never substitutes dataset images, recorded actions, or IK results.
"""
from __future__ import annotations

import argparse
import base64
import binascii
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
import json
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image, UnidentifiedImageError


MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_IMAGE_BYTES = 6 * 1024 * 1024
MAX_IMAGE_PIXELS = 4096 * 4096


def decode_image(value, camera):
    """Decode bounded PNG bytes without changing their spatial resolution."""
    if not isinstance(value, str) or not value or len(value) > (MAX_IMAGE_BYTES + 2) // 3 * 4:
        raise ValueError(f"images.{camera} must contain a bounded base64 PNG")
    try:
        data = base64.b64decode(value, validate=True)
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError("Encoded image exceeds size limit")
        with Image.open(BytesIO(data)) as image:
            if image.format != "PNG" or image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("Expected PNG with at most 4096 x 4096 pixels")
            image.load()
            return image.convert("RGB")
    except (binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError(f"Invalid {camera} PNG image: {error}") from error


def decode_request(payload):
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    if set(payload) - {"task", "state", "images", "seed"}:
        raise ValueError("Unexpected request fields")
    task = payload.get("task")
    if not isinstance(task, str) or not re.fullmatch(r"Press (?:2[4-9]|3[0-5]) floor\.", task):
        raise ValueError("task must be 'Press xx floor.' for floors 24..35")
    state = payload.get("state")
    if (not isinstance(state, list) or len(state) not in (8, 9)
            or any(type(value) not in (int, float) for value in state)):
        raise ValueError("state must be an 8D or 9D numeric vector")
    state = np.asarray(state, dtype=np.float32)
    # Reuse the exact training/inference rotation convention and validation.
    from starVLA.inference.piper_policy import prepare_state
    prepare_state(state)
    seed = payload.get("seed")
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63):
        raise ValueError("seed must be an integer in [0, 2**63), or null")
    cameras = payload.get("images")
    if not isinstance(cameras, dict) or set(cameras) != {"global", "wrist"}:
        raise ValueError("images must contain exactly global and wrist")
    return dict(task=task, state=state, seed=seed,
                global_image=decode_image(cameras["global"], "global"),
                wrist_image=decode_image(cameras["wrist"], "wrist"))


class PolicyApplication:
    def __init__(self, policy, *, loaded_seconds=0):
        self.policy = policy
        self.loaded_seconds = float(loaded_seconds)
        self.request_count = 0

    def health(self):
        checkpoint = Path(self.policy.provenance["checkpoint"])
        return dict(status="ready", checkpoint_verified=True, provenance=self.policy.provenance,
                    checkpoint_step=int(checkpoint.name.removeprefix("step_")),
                    model_sha256=self.policy.provenance["model_sha256"],
                    camera_order=["global", "wrist"], input_image_resize="RGB, bilinear square",
                    resolution=self.policy.resolution, fps=30, action_horizon=7,
                    num_inference_timesteps=int(self.policy.config.framework.action_model.num_inference_timesteps),
                    pose_frame="base_link", pose_link="gripper_tcp", xyz_units="metres",
                    rotation6d="first two matrix rows, row-major", quaternion_order="wxyz",
                    gripper_is_learned=False,
                    controller_gripper_width_m=self.policy.controller_gripper_width_m,
                    loaded_seconds=self.loaded_seconds, request_count=self.request_count)

    def predict(self, decoded):
        started = time.monotonic()
        prediction = self.policy.predict(**decoded)
        inference_seconds = time.monotonic() - started
        pose8 = np.asarray(prediction["pose8"])
        pose9 = np.asarray(prediction["raw_pose9"])
        if (pose8.shape != (7, 8) or pose9.shape != (7, 9)
                or not np.isfinite(pose8).all() or not np.isfinite(pose9).all()):
            raise RuntimeError("Policy output must be finite 7 x 8 and 7 x 9 poses")
        self.request_count += 1
        return dict(actions_pose8=pose8.tolist(), actions_pose9=pose9.tolist(),
                    inference_seconds=inference_seconds, request_index=self.request_count,
                    camera_order=["global", "wrist"], seed=decoded["seed"],
                    **{key: prediction[key] for key in ("controller_gripper_width_m", "gripper_is_learned",
                                                        "pose_frame", "pose_link", "xyz_units")})


class PolicyHTTPServer(HTTPServer):
    """Intentionally sequential: a single policy's CUDA/RNG state is shared."""
    allow_reuse_address = True

    def __init__(self, address, application):
        if address[0] != "127.0.0.1":
            raise ValueError("The inference service must bind to 127.0.0.1; use SSH forwarding")
        self.application = application
        super().__init__(address, PolicyHandler)

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(30)
        return connection, address


class PolicyHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # One structured line per request. Image payloads never enter logs.
        print(json.dumps(dict(kind="http", message=format % args)), flush=True)

    def respond(self, status, payload):
        encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path != "/health":
            self.respond(404, dict(error="Unknown endpoint"))
            return
        self.respond(200, self.server.application.health())

    def do_POST(self):
        if self.path != "/predict":
            self.respond(404, dict(error="Unknown endpoint"))
            return
        started = time.monotonic()
        try:
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Content-Type must be application/json")
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("Transfer-Encoding is unsupported; send Content-Length")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError(f"Content-Length must be 1..{MAX_REQUEST_BYTES} bytes")
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("Incomplete request body")
            decoded = decode_request(json.loads(body))
        except (ValueError, TypeError, UnicodeDecodeError, TimeoutError) as error:
            self.respond(400, dict(error=str(error), error_type=type(error).__name__))
            return
        try:
            result = self.server.application.predict(decoded)
        except Exception as error:
            self.log_message("Prediction failed: %s: %s", type(error).__name__, str(error))
            self.respond(500, dict(error=str(error), error_type=type(error).__name__))
            return
        result["seconds"] = time.monotonic() - started
        self.respond(200, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--base-vlm", type=Path)
    parser.add_argument("--base-encoder", type=Path)
    parser.add_argument("--controller-gripper-width-m", type=float, default=.008)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be 1..65535")
    from starVLA.inference.piper_policy import PiperPolicy, sha256
    started = time.monotonic()
    policy = PiperPolicy(args.checkpoint, device=args.device, base_vlm=args.base_vlm,
                         base_encoder=args.base_encoder,
                         controller_gripper_width_m=args.controller_gripper_width_m)
    application = PolicyApplication(policy, loaded_seconds=time.monotonic() - started)
    with PolicyHTTPServer(("127.0.0.1", args.port), application) as server:
        print(json.dumps(dict(kind="ready", address=f"http://127.0.0.1:{args.port}",
                              service_code_sha256=sha256(__file__), **application.health())), flush=True)
        try:
            server.serve_forever(poll_interval=.25)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
