"""Tiled image extraction preserves camera order and owns each frame's memory."""
import numpy as np
import pytest

from pressb.sensors.tiled import split_tiled_rgb


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("wrapped", [False, True])
def test_partial_last_row_keeps_camera_order_and_returns_independent_rgb(channels, wrapped):
    width, height = 3, 2
    buffer = np.full((2 * height, 3 * width, channels), 250, dtype=np.uint8)
    for index in range(5):
        row, column = divmod(index, 3)
        buffer[row * height:(row + 1) * height,
               column * width:(column + 1) * width, :3] = [index, index + 10, index + 20]
    frames = split_tiled_rgb({"data": buffer} if wrapped else buffer, 5, (width, height))
    assert len(frames) == 5
    for index, frame in enumerate(frames):
        np.testing.assert_array_equal(frame, np.broadcast_to([index, index + 10, index + 20], (height, width, 3)))
        assert frame.flags.c_contiguous
        assert not np.shares_memory(frame, buffer)
    buffer[:] = 0
    assert frames[4][0, 0].tolist() == [4, 14, 24]
    frames[0][:] = 99
    assert frames[1][0, 0].tolist() == [1, 11, 21]


@pytest.mark.parametrize("buffer,count,resolution", [
    (np.zeros((2, 3, 3), dtype=np.uint8), 0, (3, 2)),
    (np.zeros((2, 3, 3), dtype=np.uint8), 1, (0, 2)),
    (np.zeros((2, 3, 3), dtype=np.float32), 1, (3, 2)),
    (np.zeros((2, 3, 2), dtype=np.uint8), 1, (3, 2)),
    (np.zeros((2, 3, 3), dtype=np.uint8), 2, (3, 2)),
])
def test_invalid_tiled_buffer_is_rejected(buffer, count, resolution):
    with pytest.raises(ValueError):
        split_tiled_rgb(buffer, count, resolution)
