import base64
import copy
import hashlib
from io import BytesIO
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import numpy as np
from PIL import Image
import torch
from omegaconf import OmegaConf

from starVLA.inference.table_buttons_joint6_policy import (
    FIXED_GRIPPER_FULL_WIDTH_M, MODEL_ID, MODEL_GROUPS, NATIVE_SCHEMA, TableButtonsJoint6Policy,
    expand_joint6_actions, load_metadata, validate_normalization,
)
from starVLA.inference.table_buttons_policy import InvalidPolicyAction, seeded_noise


def normalization():
    names = [f"joint_{i}.pos" for i in range(1, 7)]
    state = dict(kind="fixed_joint7_scale", version=1, feature_names=names + ["gripper.pos"],
                 input_units=["degree"]*6 + ["cm"], scale=[180.]*6 + [8.], offset=[0.]*7,
                 clip=False, gripper_representation="full_width")
    action = dict(kind="fixed_joint6_scale", version=1, feature_names=names, input_units=["degree"]*6,
                  scale=[180.]*6, offset=[0.]*6, clip=False)
    return dict(kind="separate_joint7_state_joint6_action", version=1, state=state, action=action,
                fixed_gripper_full_width_m=FIXED_GRIPPER_FULL_WIDTH_M)


def fixture(root, model_id=MODEL_ID):
    root = Path(root)
    best = root / "checkpoints/best"
    best.mkdir(parents=True)
    (best / "model.pt").write_bytes(b"metadata-only fixture, not a model")
    provenance = dict(dataset_sha256="dataset", split_sha256="split", training_fingerprint="training")
    norm = normalization()
    schema = dict(schema=NATIVE_SCHEMA, state_dim=7, action_dim=6, action_horizon=7, cameras=["up", "wrist"],
                  fps=30, normalization=norm, fixed_gripper_full_width_m=FIXED_GRIPPER_FULL_WIDTH_M,
                  dataset_group=MODEL_GROUPS[model_id], **provenance)
    manifest = dict(schema=NATIVE_SCHEMA, complete=True, step=10, model_bytes=(best / "model.pt").stat().st_size, **provenance)
    config = dict(run_id=model_id, framework=dict(action_model=dict(state_dim=7, action_dim=6, action_horizon=7,
                  num_inference_timesteps=4)), datasets=dict(vla_data=dict(resolution_size=224,
                  data_root_dir=f"/not-mounted/training-data/{MODEL_GROUPS[model_id]}")))
    OmegaConf.save(OmegaConf.create(config), root / "config.yaml")
    for name, value in (("normalization.json", norm), ("action_schema.json", schema), ("checkpoints/best/manifest.json", manifest)):
        (root / name).write_text(json.dumps(value))
    if model_id != MODEL_ID:
        (root / "training_contract.json").write_text(json.dumps(dict(contract=dict(
            source_episode_count=1000, split_counts=dict(train=900,val=100)))))
    return root


class PolicyTests(unittest.TestCase):
    def test_all_red1000_ids_keep_correct_spacing_without_source_dataset(self):
        for model_id, group in MODEL_GROUPS.items():
            with self.subTest(model_id=model_id), tempfile.TemporaryDirectory() as root:
                run = fixture(root, model_id)
                _, info = load_metadata(run)
                self.assertEqual(info['model_id'], model_id)
                self.assertEqual(info['spacing'], group.split('_')[0])
                self.assertEqual(info['source_dataset_group'], group)
                self.assertEqual(info['source_episode_count'], 100 if model_id == MODEL_ID else 1000)
                self.assertEqual(info['allowed_instructions'], ['Press the red button.'])
                self.assertEqual(info['native_action_dim'], 6)
                self.assertEqual(info['fixed_gripper_full_width_m'], FIXED_GRIPPER_FULL_WIDTH_M)
                self.assertFalse(torch.cuda.is_initialized())

    def test_expanded_metadata_rejects_wrong_dataset_contract_or_explicit_model_id(self):
        with tempfile.TemporaryDirectory() as root:
            run = fixture(root, 'far_red_joint6_1000')
            with self.assertRaises(ValueError): load_metadata(run, MODEL_ID)
            contract = run/'training_contract.json'
            contract.write_text(json.dumps(dict(contract=dict(source_episode_count=100, split_counts=dict(train=90,val=10)))))
            with self.assertRaisesRegex(ValueError, '1000-episode'): load_metadata(run)
            contract.write_text(json.dumps(dict(contract=dict(source_episode_count=1000, split_counts=dict(train=900,val=100)))))
            schema = json.loads((run/'action_schema.json').read_text()); schema['dataset_group']='near_red'
            (run/'action_schema.json').write_text(json.dumps(schema))
            with self.assertRaisesRegex(ValueError, 'dataset group'): load_metadata(run)

    def test_native_action_has_no_gripper_noise_and_wire_appends_full_width(self):
        arm = np.linspace(-2, 2, 2*7*6, dtype=np.float32).reshape(2,7,6)
        norm, physical = expand_joint6_actions(arm, FIXED_GRIPPER_FULL_WIDTH_M)
        np.testing.assert_array_equal(norm[..., :6], arm)
        np.testing.assert_array_equal(physical[..., :6], arm*180)
        self.assertEqual(norm.shape, (2,7,7))
        self.assertTrue(np.all(physical[..., 6] == np.float32(.02)))
        np.testing.assert_allclose(physical[..., 6]*.01, FIXED_GRIPPER_FULL_WIDTH_M, atol=1e-11, rtol=0)
        np.testing.assert_allclose(physical, norm*np.array([180.]*6+[8.]), atol=2e-5, rtol=2e-6)

    def test_wrong_native_dimensions_and_nonfinite_are_typed_model_failures(self):
        for value in (np.zeros((1,7,7)), np.zeros((7,6)), np.full((1,7,6), np.nan), np.full((1,7,6), np.inf)):
            with self.assertRaises(InvalidPolicyAction):
                expand_joint6_actions(value, FIXED_GRIPPER_FULL_WIDTH_M)

    def test_reject_width_units_or_wrong_normalization(self):
        self.assertEqual(validate_normalization(normalization()), FIXED_GRIPPER_FULL_WIDTH_M)
        for width in (.02, .002, -.0002, float("nan"), True):
            spec = normalization(); spec["fixed_gripper_full_width_m"] = width
            with self.assertRaises(ValueError): validate_normalization(spec)
        for key, value in (("scale", [180.]*7), ("input_units", ["rad"]*6), ("clip", True)):
            spec = normalization(); spec["action"][key] = value
            with self.assertRaises(ValueError): validate_normalization(spec)

    def test_metadata_reads_native_schema_without_model_or_cuda_loading(self):
        with tempfile.TemporaryDirectory() as root:
            run = fixture(root)
            _, info = load_metadata(run)
            self.assertEqual(info["native_action_dim"], 6)
            self.assertEqual(info["wire_action_dim"], 7)
            self.assertEqual(info["state_dim"], 7)
            self.assertIsNone(info["model_sha256"])
            self.assertFalse(info["gripper_learned"])
            self.assertFalse(torch.cuda.is_initialized())
            schema = json.loads((run / "action_schema.json").read_text()); schema["action_dim"] = 7
            (run / "action_schema.json").write_text(json.dumps(schema))
            with self.assertRaises(ValueError): load_metadata(run)

    def test_reject_checkpoint_provenance_change(self):
        with tempfile.TemporaryDirectory() as root:
            run = fixture(root)
            path = run / "checkpoints/best/manifest.json"
            value = json.loads(path.read_text()); value["dataset_sha256"] = "other"
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError): load_metadata(run)

    def test_six_channel_noise_and_per_sample_batch_seeding(self):
        from types import SimpleNamespace
        head = SimpleNamespace(action_horizon=7, action_dim=6, dtype=torch.float32)
        noise = seeded_noise(head, 2, [12, 42], torch.device("cpu"))
        self.assertEqual(tuple(noise.shape), (2,7,6))
        for i, seed in enumerate((12,42)):
            torch.manual_seed(seed)
            self.assertTrue(torch.equal(noise[i:i+1], torch.randn((1,7,6))))

    def fake_policy(self):
        policy = object.__new__(TableButtonsJoint6Policy)
        policy.identity = dict(model_id=MODEL_ID, model_sha256="test-hash", allowed_instructions=["Press the red button."],
                               native_action_dim=6, state_dim=7, action_dim=7, fixed_gripper_full_width_m=FIXED_GRIPPER_FULL_WIDTH_M,
                               action_units=["degree"]*6+["full_width_cm"])
        policy.resolution, policy.device = 224, "cpu"
        def prepared(images, states, instructions, seeds, *, return_noise=False):
            noise = np.zeros((len(images),7,6), dtype=np.float32)
            result = (np.full((len(images),7,6), .125, dtype=np.float32),
                      [hashlib.sha256(row.tobytes()).hexdigest() for row in noise])
            return (*result, noise) if return_noise else result
        policy.predict_prepared = prepared
        return policy

    def test_prediction_uses_measured_state_and_exports_separate_native_actions(self):
        policy = self.fake_policy()
        sample = dict(episode_id="ep", request_id=0, instruction="Press the red button.",
                      state=[0]*6+[.021], images=[Image.new("RGB",(640,480))]*2, noise_seed=5)
        with patch("torch.cuda.synchronize"):
            result = policy.predict([sample])
        p = result["predictions"][0]
        self.assertEqual(np.asarray(p["native_actions"]).shape, (7,6))
        self.assertEqual(np.asarray(p["actions"]).shape, (7,7))
        noise = np.asarray(p["initial_noise"], dtype=np.float32)
        self.assertEqual(noise.shape, (7,6))
        self.assertEqual(hashlib.sha256(noise.tobytes()).hexdigest(), p["initial_noise_sha256"])
        self.assertAlmostEqual(p["normalized_state"][6], .021/8, places=9)
        self.assertNotEqual(p["normalized_state"][6], p["normalized_actions"][0][6])
        self.assertTrue(np.all(np.asarray(p["actions"])[:,6] == float(np.float32(.02))))

    def test_joint7_http_protocol_reset_and_prediction_unchanged(self):
        path = Path(__file__).resolve().parents[1]/"scripts/serve_table_buttons_joint6_policy.py"
        spec = importlib.util.spec_from_file_location("joint6_service", path)
        service = importlib.util.module_from_spec(spec); spec.loader.exec_module(service)
        policy = self.fake_policy()
        app = object.__new__(service.Application)
        app.spacing, app.device, app.checkpoint = "medium", "cpu", None
        app.registry, app.identities, app.loaded = {MODEL_ID: Path("unused")}, {MODEL_ID: policy.identity}, {MODEL_ID: policy}
        app.sessions, app.request_count = {}, 0
        stream = BytesIO(); Image.new("RGB", (640,480)).save(stream, format="PNG")
        png = base64.b64encode(stream.getvalue()).decode()
        sample = dict(episode_id="ep", request_id=0, instruction="Press the red button.", state=[0]*6+[.021],
                      images={"up":png,"wrist":png}, noise_seed=42)
        with service.Server(0, app) as server, patch("torch.cuda.synchronize"):
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval":.01}, daemon=True); thread.start()
            def post(route, body):
                request=Request(f"http://127.0.0.1:{server.server_port}{route}",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
                with urlopen(request,timeout=5) as response: return json.load(response)
            try:
                with self.assertRaises(HTTPError) as error: post("/predict",dict(model_id=MODEL_ID,samples=[sample]))
                self.assertEqual(error.exception.code,400)
                post("/reset",dict(model_id=MODEL_ID,episode_id="ep"))
                output=post("/predict",dict(model_id=MODEL_ID,samples=[sample]))
                self.assertEqual(output["schema"],"pressb.table_buttons.joint7_policy.v1")
                self.assertEqual(output["protocol_version"],1)
                self.assertEqual(output["model"]["native_action_dim"],6)
                self.assertEqual(np.asarray(output["predictions"][0]["actions"]).shape,(7,7))
            finally:
                server.shutdown();thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
