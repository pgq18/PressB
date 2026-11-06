"""Independent joint6 policy server; preserve the existing joint7 HTTP wire API.

--check-metadata performs no model/GPU loading and never opens a listening socket.
The original twelve-model server is imported for protocol handling but unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "scripts"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from serve_table_buttons_policies import Application as Joint7Application, Server
from starVLA.inference.table_buttons_joint6_policy import MODEL_GROUPS, NATIVE_SCHEMA, TableButtonsJoint6Policy, load_metadata


class Application(Joint7Application):
    def __init__(self, run, model_id=None, device="cuda:0", checkpoint=None):
        identity = load_metadata(run, model_id, checkpoint)[1]
        model_id = identity["model_id"]
        self.spacing, self.device = identity["spacing"], device
        self.registry, self.checkpoint = {model_id: Path(run)}, checkpoint
        self.identities = {model_id: identity}
        self.loaded, self.sessions, self.request_count = {}, {}, 0

    def load(self, model_id):
        if model_id not in self.registry:
            raise ValueError("Unknown model_id for this independent joint6 server")
        if model_id not in self.loaded:
            print(json.dumps(dict(event="loading_model", model_id=model_id, native_action_dim=6)), flush=True)
            policy = TableButtonsJoint6Policy(self.registry[model_id], model_id, self.device, self.checkpoint)
            self.loaded[model_id] = policy
            self.identities[model_id] = dict(policy.identity)
            print(json.dumps(dict(event="model_ready", **policy.identity)), flush=True)
        return self.loaded[model_id]

    def health(self):
        return dict(super().health(), native_schema=NATIVE_SCHEMA, native_action_dim=6,
                    wire_action_dim=7, state_dim=7, gripper_learned=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Completed run metadata and saved processor")
    parser.add_argument("--model-id", choices=list(MODEL_GROUPS), help="Defaults to the saved config run_id")
    parser.add_argument("--checkpoint", type=Path, help="Explicit checkpoint directory; defaults to run/checkpoints/best")
    parser.add_argument("--port", type=int, default=18113)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--check-metadata", action="store_true")
    args = parser.parse_args()
    app = Application(args.run, args.model_id, args.device, args.checkpoint)
    if args.check_metadata:
        print(json.dumps(dict(metadata_only=True, gpu_used=False, **app.health()), indent=2))
        return
    if args.preload:
        app.load(args.model_id)
    with Server(args.port, app) as server:
        print(json.dumps(dict(event="ready", port=args.port, **app.health())), flush=True)
        server.serve_forever(poll_interval=.25)


if __name__ == "__main__":
    main()
