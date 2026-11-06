"""CPU HTTP contract tests with a fake policy, without sockets or model loading."""
import base64
import contextlib
from email.message import Message
import importlib.util
from io import BytesIO, StringIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.serve_piper_policy import (MAX_REQUEST_BYTES, PolicyApplication, PolicyHandler,
                                       PolicyHTTPServer)
from starVLA.inference import piper_policy

# The numeric implementation is real; bypass the unrelated dataloader package
# initializer so this CPU-only test does not require Accelerate or model deps.
spec = importlib.util.spec_from_file_location("piper_http_test_poses", Path(__file__).resolve().parents[1] /
                                            "starVLA/dataloader/piper_lerobot.py")
poses = importlib.util.module_from_spec(spec)
spec.loader.exec_module(poses)


def encode_image(color, format="PNG"):
    image = Image.new("RGB", (64, 48), color)
    stream = BytesIO()
    image.save(stream, format=format)
    return base64.b64encode(stream.getvalue()).decode()


def good_request():
    return dict(task="Press 24 floor.", state=[.1, .2, .3, 1., 0., 0., 0., .008], seed=42,
                images={"global": encode_image((200, 10, 0)), "wrist": encode_image((0, 10, 200))})


class FakePolicy:
    def __init__(self):
        self.provenance = dict(checkpoint="/run/checkpoints/step_002000", model_sha256="a" * 64)
        self.resolution = 224
        self.controller_gripper_width_m = .008
        self.config = SimpleNamespace(framework=SimpleNamespace(action_model=SimpleNamespace(num_inference_timesteps=4)))
        self.calls = []
        self.fail = False

    def predict(self, **arguments):
        self.calls.append(arguments)
        if self.fail:
            raise RuntimeError("injected inference failure")
        from starVLA.inference.piper_policy import prepare_state, pose9_to_pose8
        raw = np.repeat(prepare_state(arguments["state"])[0], 7, axis=0)
        return dict(raw_pose9=raw, pose8=pose9_to_pose8(raw), controller_gripper_width_m=.008,
                    gripper_is_learned=False, pose_frame="base_link", pose_link="gripper_tcp", xyz_units="metres")


def http(application, payload=None, *, path="/predict", method="POST", headers=None, body=None):
    handler = object.__new__(PolicyHandler)
    handler.server = SimpleNamespace(application=application)
    handler.command = method
    handler.request_version = "HTTP/1.1"
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.path = path
    handler.close_connection = False
    data = json.dumps(payload).encode() if body is None else body
    handler.headers = Message()
    for key, value in (headers or {"Content-Type": "application/json", "Content-Length": str(len(data))}).items():
        handler.headers[key] = value
    handler.rfile, handler.wfile = BytesIO(data), BytesIO()
    with contextlib.redirect_stdout(StringIO()):
        getattr(handler, f"do_{method}")()
    head, content = handler.wfile.getvalue().split(b"\r\n\r\n", 1)
    status = int(head.splitlines()[0].split()[1])
    return status, json.loads(content)


class TestPolicyServer(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(piper_policy, "_pose_helpers", return_value=poses)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.policy = FakePolicy()
        self.application = PolicyApplication(self.policy, loaded_seconds=3)

    def test_health_binds_provenance_and_training_inference_configuration(self):
        status, result = http(self.application, method="GET", path="/health")
        self.assertEqual(status, 200)
        self.assertEqual(result["checkpoint_step"], 2000)
        self.assertEqual(result["model_sha256"], "a" * 64)
        self.assertEqual(result["num_inference_timesteps"], 4)
        self.assertEqual(result["camera_order"], ["global", "wrist"])
        self.assertTrue(result["checkpoint_verified"])
        self.assertEqual(self.policy.calls, [])

    def test_http_predict_preserves_observations_state_seed_and_pose_convention(self):
        payload = good_request()
        status, result = http(self.application, payload)
        self.assertEqual(status, 200)
        self.assertEqual(len(self.policy.calls), 1)
        arguments = self.policy.calls[0]
        self.assertEqual(arguments["seed"], 42)
        self.assertEqual(arguments["task"], "Press 24 floor.")
        self.assertEqual(arguments["global_image"].size, (64, 48))
        self.assertEqual(arguments["global_image"].getpixel((0, 0)), (200, 10, 0))
        self.assertEqual(arguments["wrist_image"].getpixel((0, 0)), (0, 10, 200))
        np.testing.assert_array_equal(arguments["state"], np.asarray(payload["state"], dtype=np.float32))
        actions = np.asarray(result["actions_pose8"])
        self.assertEqual(actions.shape, (7, 8))
        self.assertEqual(np.asarray(result["actions_pose9"]).shape, (7, 9))
        np.testing.assert_array_equal(actions[:, :3], np.tile(arguments["state"][:3], (7, 1)))
        np.testing.assert_allclose(actions[:, 3:7], np.tile([1, 0, 0, 0], (7, 1)))
        np.testing.assert_allclose(actions[:, 7], .008)
        self.assertFalse(result["gripper_is_learned"])
        self.assertEqual(result["pose_frame"], "base_link")
        self.assertGreaterEqual(result["seconds"], result["inference_seconds"])

    def test_pose9_request_matches_pose8_without_changing_position(self):
        payload = good_request()
        payload["state"] = [.1, .2, .3, 1, 0, 0, 0, 1, 0]
        status, result = http(self.application, payload)
        self.assertEqual(status, 200)
        np.testing.assert_allclose(result["actions_pose9"], np.tile(payload["state"], (7, 1)))

    def test_invalid_inputs_never_invoke_model(self):
        bad = []
        for field, values in (("task", ["Press 36 floor.", "press 24", 24]),
                              ("state", [[0] * 8, [0] * 9, [float("nan")] * 8, [True] * 8, ["1"] * 8]),
                              ("seed", [-1, 2**63, True, 1.5]),
                              ("images", [{}, {"global": "bad", "wrist": "bad"},
                                          {"global": encode_image((0, 0, 0), "JPEG"),
                                           "wrist": encode_image((0, 0, 0))}])):
            for value in values:
                payload = good_request()
                payload[field] = value
                bad.append(payload)
        payload = good_request()
        payload["actions"] = [0] * 8
        bad.append(payload)
        for payload in bad:
            with self.subTest(payload=str(payload)[:120]):
                status, result = http(self.application, payload)
                self.assertEqual(status, 400)
                self.assertIn("error", result)
        self.assertEqual(self.policy.calls, [])

    def test_http_boundary_checks_and_unknown_routes(self):
        cases = [dict(body=b"not json"),
                 dict(headers={"Content-Type": "text/plain", "Content-Length": "2"}, body=b"{}"),
                 dict(headers={"Content-Type": "application/json", "Content-Length": str(MAX_REQUEST_BYTES + 1)}),
                 dict(headers={"Content-Type": "application/json", "Content-Length": "100"}, body=b"{}"),
                 dict(headers={"Content-Type": "application/json", "Transfer-Encoding": "chunked"})]
        for options in cases:
            self.assertEqual(http(self.application, **options)[0], 400)
        self.assertEqual(http(self.application, method="GET", path="/other")[0], 404)
        self.assertEqual(http(self.application, path="/other")[0], 404)
        self.assertEqual(self.policy.calls, [])

    def test_failed_inference_is_explicit_and_service_remains_usable(self):
        self.policy.fail = True
        status, result = http(self.application, good_request())
        self.assertEqual(status, 500)
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertEqual(self.application.request_count, 0)
        self.policy.fail = False
        self.assertEqual(http(self.application, good_request())[0], 200)
        self.assertEqual(self.application.request_count, 1)

    def test_service_rejects_non_loopback_bind(self):
        with self.assertRaisesRegex(ValueError, "127.0.0.1"):
            PolicyHTTPServer(("0.0.0.0", 0), self.application)


if __name__ == "__main__":
    unittest.main()
