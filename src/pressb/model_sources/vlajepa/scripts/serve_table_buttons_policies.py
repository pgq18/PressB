"""One-spacing HTTP joint7 inference server, reachable through SSH forwarding."""
from __future__ import annotations
import argparse
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from PIL import Image
from starVLA.inference.table_buttons_policy import CAMERAS, COLORS, SCHEMA, TableButtonsPolicy, load_metadata, InvalidPolicyAction


def decode_sample(sample):
    if not isinstance(sample, dict) or set(sample) != {"episode_id", "request_id", "instruction", "state", "images", "noise_seed"}:
        raise ValueError("Sample fields must be episode_id, request_id, instruction, state, images, noise_seed")
    for field in ("episode_id", "request_id"):
        if not isinstance(sample[field], (str, int)) or isinstance(sample[field], bool) or len(str(sample[field])) > 200:
            raise ValueError(f"Invalid {field}")
    if type(sample["noise_seed"]) is not int or not 0 <= sample["noise_seed"] < 2**63:
        raise ValueError("noise_seed must be an integer in [0, 2**63)")
    if not isinstance(sample["instruction"], str):
        raise ValueError("instruction must be a string")
    cameras = sample["images"]
    if not isinstance(cameras, dict) or set(cameras) != set(CAMERAS):
        raise ValueError("images must contain exactly up and wrist")
    images = []
    for camera in CAMERAS:
        text = cameras[camera]
        if not isinstance(text, str) or not 0 < len(text) < 6 * 1024**2:
            raise ValueError("Each camera must contain bounded base64 PNG")
        data = base64.b64decode(text, validate=True)
        with Image.open(BytesIO(data)) as image:
            if image.format != "PNG" or image.size != (640, 480) or image.mode != "RGB":
                raise ValueError("Each camera must be an RGB 640x480 PNG")
            image.load()
            images.append(image.copy())
    return dict(sample, images=images)


class Application:
    def __init__(self, spacing, single_root, multitask_root, device):
        self.spacing, self.device = spacing, device
        self.registry = {f"{spacing}_{color}": Path(single_root) / f"{spacing}_{color}" for color in COLORS}
        self.registry[f"{spacing}_multitask"] = Path(multitask_root) / f"{spacing}_multitask"
        self.identities = {name: load_metadata(path, name)[1] for name, path in self.registry.items()}
        self.loaded, self.sessions, self.request_count = {}, {}, 0

    def load(self, model_id):
        if model_id not in self.registry:
            raise ValueError(f"Unknown model_id for {self.spacing}: {model_id}")
        if model_id not in self.loaded:
            print(json.dumps(dict(event="loading_model", model_id=model_id)), flush=True)
            policy = TableButtonsPolicy(self.registry[model_id], model_id, self.device)
            self.loaded[model_id] = policy
            print(json.dumps(dict(event="model_ready", **policy.identity)), flush=True)
        return self.loaded[model_id]

    def health(self):
        return dict(status="ready", schema=SCHEMA, protocol_version=1, spacing=self.spacing,
                    device=self.device, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), available_models=self.identities, loaded_models={key: value.identity for key, value in self.loaded.items()},
                    request_count=self.request_count, camera_order=list(CAMERAS), action_units=["degree"] * 6 + ["full_width_cm"],
                    action_horizon=7, input_encoding="RGB PNG base64", recurrent_state="none", smoothing="none")

    def dispatch(self, route, payload):
        if not isinstance(payload, dict):
            raise ValueError("Expected JSON object")
        model_id = payload.get("model_id")
        if model_id not in self.registry:
            raise ValueError("Explicit model_id must belong to this spacing server")
        if route == "/load":
            return dict(model=self.load(model_id).identity, status="loaded")
        if route == "/reset":
            episode = payload.get("episode_id")
            if not isinstance(episode, (int, str)) or isinstance(episode, bool):
                raise ValueError("reset requires episode_id")
            policy = self.load(model_id)
            self.sessions[(model_id, str(episode))] = 0
            return dict(model=policy.identity, episode_id=episode, status="reset", recurrent_state="none")
        if route != "/predict":
            raise ValueError("Unknown route")
        rows = payload.get("samples")
        if not isinstance(rows, list) or not 1 <= len(rows) <= 16:
            raise ValueError("samples must have 1..16 entries")
        samples = [decode_sample(row) for row in rows]
        for sample in samples:
            if (model_id, str(sample["episode_id"])) not in self.sessions:
                raise ValueError("Call /reset for this model_id and episode_id before predicting")
        result = self.load(model_id).predict(samples)
        self.request_count += 1
        result["request_index"] = self.request_count
        for sample in samples:
            self.sessions[(model_id, str(sample["episode_id"]))] += 1
        return result


class Server(HTTPServer):
    allow_reuse_address = True
    def __init__(self, port, app):
        self.application = app
        super().__init__(("127.0.0.1", port), Handler)
    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(120)
        return connection, address


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass
    def respond(self, code, value):
        data = json.dumps(dict(value, schema=SCHEMA, protocol_version=1), allow_nan=False, separators=(",", ":")).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass
    def do_GET(self):
        self.respond(200, self.server.application.health()) if self.path == "/health" else self.respond(404, dict(error="Unknown route"))
    def do_POST(self):
        started = time.monotonic()
        try:
            if self.path not in ("/load", "/reset", "/predict"):
                self.respond(404, dict(error="Unknown route"))
                return
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 64 * 1024**2 or self.headers.get_content_type() != "application/json":
                raise ValueError("Expected bounded application/json request")
            data = self.rfile.read(size)
            if len(data) != size:
                raise ValueError("Incomplete request body")
            payload = json.loads(data, parse_constant=lambda text: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
            result = self.server.application.dispatch(self.path, payload)
            result["total_seconds"] = time.monotonic() - started
            self.respond(200, result)
        except InvalidPolicyAction as error:
            self.respond(422, dict(error=str(error), error_kind="invalid_policy_action"))
        except (ValueError, KeyError, TypeError, OSError) as error:
            self.respond(400, dict(error=f"{type(error).__name__}: {error}"))
        except Exception as error:
            print(json.dumps(dict(event="request_failed", route=self.path, error=f"{type(error).__name__}: {error}")), flush=True)
            self.respond(500, dict(error=f"{type(error).__name__}: {error}"))


def main():
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spacing", choices=("near", "medium", "far"), required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--single-root", type=Path, default=repository / "runs/table_buttons_joint7_20261023")
    parser.add_argument("--multitask-root", type=Path, default=repository / "runs/table_buttons_multitask_joint7_20261023")
    parser.add_argument("--preload-all", action="store_true")
    args = parser.parse_args()
    app = Application(args.spacing, args.single_root, args.multitask_root, args.device)
    if args.preload_all:
        for model_id in app.registry:
            app.load(model_id)
    with Server(args.port, app) as server:
        print(json.dumps(dict(event="ready", port=args.port, **app.health())), flush=True)
        server.serve_forever(poll_interval=.25)


if __name__ == "__main__":
    main()
