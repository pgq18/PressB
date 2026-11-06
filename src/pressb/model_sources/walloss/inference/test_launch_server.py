"""Launcher regressions; no checkpoint loading, simulation, or GPU work."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
import yaml

import launch_server


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "config.yml"
        self.original = {"norm_stats_path": "training-stats.json", "model_type": "wall-oss",
                         "data": {"lerobot_config": {"repo_id": "lerobot/piper"}}}
        self.path.write_text(yaml.safe_dump(self.original))

    def config(self, stats=None):
        return launch_server.ModelConfig(
            model_path=self.directory.name, train_config_path=str(self.path),
            norm_stats_path=stats, state_dim=7, action_dim=7)

    def test_stats_override_reaches_model_without_changing_training_file(self):
        args = launch_server.Args(model_config=self.config("local-stats.json"))
        with patch.object(launch_server, "WallXPolicy") as policy:
            launch_server.create_policy(args)
        self.assertEqual(policy.call_args.kwargs["train_config"]["norm_stats_path"], "local-stats.json")
        self.assertEqual(yaml.safe_load(self.path.read_text()), self.original)

    def test_absent_override_keeps_training_normalizer(self):
        with patch.object(launch_server, "WallXPolicy") as policy:
            launch_server.create_policy(launch_server.Args(model_config=self.config()))
        self.assertEqual(policy.call_args.kwargs["train_config"]["norm_stats_path"], "training-stats.json")

    def test_main_binds_requested_host_and_shares_stats_override(self):
        args = launch_server.Args(model_config=self.config("local-stats.json"), host="127.0.0.1", port=8094)
        stat = Mock(min=torch.zeros(7), delta=torch.ones(7))
        policy = Mock(metadata={"action_dim": 7})
        with patch.object(launch_server, "load_norm_stats", return_value={"action": stat, "state": stat}) as load, \
                patch.object(launch_server, "create_policy", return_value=policy), \
                patch.object(launch_server.socket, "gethostbyname", return_value="127.0.0.1"), \
                patch.object(launch_server.websocket_policy_server, "WebsocketPolicyServer") as server:
            launch_server.main(args)
        load.assert_called_once_with("local-stats.json", "lerobot/piper")
        self.assertEqual(server.call_args.kwargs["host"], "127.0.0.1")
        self.assertEqual(server.call_args.kwargs["port"], 8094)
        server.return_value.serve_forever.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
