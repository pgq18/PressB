import threading

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pressb.online_rl.protocol import (encode_image, decode_image, pose8_to_pose9, pose9_to_pose8,
                                       initial_noise, residual_actions)
from pressb.online_rl.rpc import RPCServer, RPCClient, RPCError


def test_wire_preserves_images_and_row_rotation_convention():
    rng = np.random.default_rng(12)
    rgb = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    assert np.array_equal(rgb, decode_image(encode_image(rgb)))
    r = Rotation.from_euler("xyz", [.3, -.4, 1.2])
    pose8 = np.r_[.2, -.1, .4, r.as_quat()[[3, 0, 1, 2]], .008]
    pose9 = pose8_to_pose9(pose8)
    np.testing.assert_allclose(pose9[3:], r.as_matrix()[:2].reshape(-1), atol=1e-15)
    np.testing.assert_allclose(pose9_to_pose8(pose9), pose8, atol=1e-15)
    with pytest.raises(ValueError, match="Degenerate"):
        pose9_to_pose8(np.zeros(9))


def test_modulation_preserves_xyz_units_and_noise_periodicity():
    base = np.tile([.2, .3, .4, 1, 0, 0, 0, 1, 0], (7, 1))
    action = residual_actions(base, np.ones(63), [.03] * 3 + [.1] * 6)
    np.testing.assert_allclose(action[0, :3], [.23, .33, .43])
    np.testing.assert_allclose(pose9_to_pose8(action)[:, :3], action[:, :3])
    u = np.arange(9) / 9
    np.testing.assert_allclose(initial_noise(u), np.tile(u * 1.5, (7, 1)))
    with pytest.raises(ValueError, match="divide"):
        initial_noise(np.zeros(18), 2)


def test_rpc_real_port_auth_validation_and_single_execution():
    calls = []
    server = RPCServer(("127.0.0.1", 0), {"/health": lambda _: {"ready": True},
        "/step": lambda p: calls.append(p["x"]) or {"result": p["x"]}}, token="test-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    address = "http://127.0.0.1:" + str(server.server_port)
    try:
        with pytest.raises(RPCError, match="401"):
            RPCClient(address).call("/health")
        client = RPCClient(address, token="test-token")
        assert client.call("/health")["ready"]
        assert client.call("/step", {"x": 3})["result"] == 3
        with pytest.raises(RPCError, match="400"):
            client.call("/step", {})
        assert calls == [3]
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
