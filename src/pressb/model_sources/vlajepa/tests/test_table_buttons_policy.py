import base64
from io import BytesIO
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace
import json
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import numpy as np
from PIL import Image
import torch

from starVLA.inference.table_buttons_policy import normalize_state, prepare_images, seeded_noise, TableButtonsPolicy, InvalidPolicyAction


class PolicyTests(unittest.TestCase):
    def test_physical_joint_units_and_gripper_width(self):
        np.testing.assert_array_equal(normalize_state([180, -90, 45, 0, 18, -18, 8]), np.array([1, -.5, .25, 0, .1, -.1, 1], dtype=np.float32))
        with self.assertRaises(ValueError):
            normalize_state([0] * 9)

    def test_exact_training_resize(self):
        image = Image.fromarray(np.random.default_rng(2).integers(0, 256, (480, 640, 3), dtype=np.uint8))
        prepared = prepare_images([image, image])
        expected = image.resize((224, 224), Image.Resampling.BILINEAR)
        np.testing.assert_array_equal(prepared[0], expected)
        with self.assertRaises(ValueError):
            prepare_images([image.resize((256, 256)), image])

    def test_seeded_noise_independent_of_batch_order_and_global_rng(self):
        head = SimpleNamespace(action_horizon=7, action_dim=7, dtype=torch.float32)
        seeds = [12, 42]
        torch.manual_seed(111)
        old = torch.get_rng_state().clone()
        generated = seeded_noise(head, 2, seeds, torch.device("cpu"))
        self.assertTrue(torch.equal(old, torch.get_rng_state()))
        for i, seed in enumerate(seeds):
            torch.manual_seed(seed)
            self.assertTrue(torch.equal(generated[i:i+1], torch.randn((1, 7, 7))))
        reversed_noise = seeded_noise(head, 2, seeds[::-1], torch.device("cpu"))
        self.assertTrue(torch.equal(generated, reversed_noise.flip(0)))

    def test_single_task_rejects_other_color_before_inference(self):
        policy = object.__new__(TableButtonsPolicy)
        policy.identity = dict(model_id="near_red", allowed_instructions=["Press the red button."])
        policy.resolution = 224
        with self.assertRaises(ValueError):
            policy.prepare([dict(instruction="Press the blue button.")])

    def test_invalid_model_actions_are_typed_episode_failure(self):
        spec = importlib.util.spec_from_file_location("service", Path(__file__).resolve().parents[1] / "scripts/serve_table_buttons_policies.py")
        service = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(service)
        class Application:
            def dispatch(self, route, payload):
                raise InvalidPolicyAction("Nonfinite model output")
        with service.Server(0, Application()) as server:
            thread = threading.Thread(target=server.handle_request)
            thread.start()
            request = Request(f"http://127.0.0.1:{server.server_port}/predict", data=b"{}",
                              headers={"Content-Type": "application/json"})
            with self.assertRaises(HTTPError) as caught:
                urlopen(request, timeout=5)
            self.assertEqual(caught.exception.code, 422)
            self.assertEqual(json.load(caught.exception)["error_kind"], "invalid_policy_action")
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
