"""NPU compatibility contracts that can be verified without Ascend hardware."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from pressb.models import pi05


def test_npu_config_construction_is_scoped_and_preserves_saved_device(tmp_path, monkeypatch):
    pytest.importorskip("lerobot")
    import draccus
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.utils.utils import is_torch_device_available

    devices = []
    monkeypatch.setattr(pi05, "initialize_pi05_device", lambda value: devices.append(value) or value)
    policy = pi05.parse_pi05_config({"device": "npu:3", "use_amp": True, "dtype": "bfloat16"})
    assert policy.device == "npu:3"
    assert policy.use_amp is True
    assert devices == ["npu:3"]
    (tmp_path / "config.json").write_text(json.dumps(draccus.encode(policy)))
    restored = pi05.load_pi05_config(tmp_path)
    assert restored.device == "npu:3"
    assert restored.use_amp is True
    assert pi05.load_pi05_config(tmp_path, device="cpu").device == "cpu"
    assert isinstance(pi05.parse_pi05_config({"device": "cpu"}), PI05Config)
    # The upstream device validator is deliberately not monkeypatched.
    with pytest.raises(ValueError, match="Unknown device"):
        is_torch_device_available("npu:3")


def test_npu_request_reports_unavailable_hardware_without_cpu_fallback(monkeypatch):
    import sys
    import torch

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_available=lambda: False), raising=False)
    with pytest.raises(RuntimeError, match="no Ascend device"):
        pi05.initialize_pi05_device("npu:0")


def test_npu_suffix_matches_native_forward_and_parameter_gradients():
    pytest.importorskip("lerobot")
    import torch
    from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch
    from torch.utils.checkpoint import checkpoint

    class Suffix(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(min_period=4e-3, max_period=4., chunk_size=7)
            self.action_in_proj = torch.nn.Linear(32, 128)
            self.time_mlp_in = torch.nn.Linear(128, 128)
            self.time_mlp_out = torch.nn.Linear(128, 128)

        def _apply_checkpoint(self, function, value):
            return checkpoint(function, value, use_reentrant=False)

    torch.manual_seed(718)
    native = Suffix()
    compatible = deepcopy(native)
    actions = torch.randn(8, 7, 32, requires_grad=True)
    other_actions = actions.detach().clone().requires_grad_(True)
    times = torch.linspace(.001, .999, 8)
    expected = PI05Pytorch.embed_suffix(native, actions, times)
    actual = pi05._embed_suffix_npu(compatible, other_actions, times)
    for left, right in zip(expected, actual, strict=True):
        torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-4)
    (expected[0].square().mean() + expected[3].square().mean()).backward()
    (actual[0].square().mean() + actual[3].square().mean()).backward()
    torch.testing.assert_close(actions.grad, other_actions.grad, atol=0, rtol=0)
    for left, right in zip(native.parameters(), compatible.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, atol=1e-7, rtol=2e-3)


def test_npu_policy_adjustment_does_not_replace_upstream_or_cpu_methods(monkeypatch):
    pytest.importorskip("lerobot")
    from lerobot.policies.pi05 import modeling_pi05

    class Core:
        def embed_suffix(self):
            return "upstream"

    class Policy:
        def __init__(self, config):
            self.model = Core()

    monkeypatch.setattr(modeling_pi05, "PI05Policy", Policy)
    monkeypatch.setattr(pi05, "initialize_pi05_device", lambda value: value)
    npu = pi05.create_pi05_policy(SimpleNamespace(device="npu:0"))
    cpu = pi05.create_pi05_policy(SimpleNamespace(device="cpu"))
    assert npu.model.embed_suffix.__func__ is pi05._embed_suffix_npu
    assert cpu.model.embed_suffix() == "upstream"
    assert Core().embed_suffix() == "upstream"
