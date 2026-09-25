from types import SimpleNamespace

import numpy as np
import pytest

from vlm_nav.image_utils import image_msg_to_array, image_msg_to_rgb


def _msg(arr, encoding, step=None, big=False):
    raw = arr.tobytes()
    h, w = arr.shape[:2]
    row = len(raw) // h
    step = step or row
    if step > row:  # add per-row padding, like real drivers do
        raw = b''.join(raw[i * row:(i + 1) * row] + b'\xaa' * (step - row) for i in range(h))
    return SimpleNamespace(height=h, width=w, step=step, encoding=encoding, is_bigendian=int(big), data=raw)


RGB = np.arange(4 * 3 * 3, dtype=np.uint8).reshape(4, 3, 3)


def test_rgb8_passthrough():
    assert np.array_equal(image_msg_to_rgb(_msg(RGB, 'rgb8')), RGB)


def test_bgr8_is_swapped_to_rgb():
    assert np.array_equal(image_msg_to_rgb(_msg(RGB[:, :, ::-1].copy(), 'bgr8')), RGB)


def test_rgba8_does_not_crash_and_drops_alpha():
    rgba = np.concatenate([RGB, np.full((4, 3, 1), 255, np.uint8)], axis=2)
    assert np.array_equal(image_msg_to_rgb(_msg(rgba, 'rgba8')), RGB)


def test_bgra8():
    bgra = np.concatenate([RGB[:, :, ::-1], np.full((4, 3, 1), 7, np.uint8)], axis=2).copy()
    assert np.array_equal(image_msg_to_rgb(_msg(bgra, 'bgra8')), RGB)


def test_row_padding_uses_step():
    assert np.array_equal(image_msg_to_rgb(_msg(RGB, 'rgb8', step=3 * 3 + 5)), RGB)


def test_mono8_expands_to_three_channels():
    g = np.arange(12, dtype=np.uint8).reshape(3, 4)
    out = image_msg_to_rgb(_msg(g, 'mono8'))
    assert out.shape == (3, 4, 3) and np.array_equal(out[:, :, 1], g)


@pytest.mark.parametrize('big', [False, True])
def test_float_depth_endianness(big):
    d = np.array([[1.5, 2.5], [3.5, np.nan]], dtype='>f4' if big else '<f4')
    out = image_msg_to_array(_msg(d, '32FC1', big=big))
    assert out.dtype == np.float32 and out.shape == (2, 2)
    assert np.allclose(out[:1], [[1.5, 2.5]]) and out[1, 0] == pytest.approx(3.5) and np.isnan(out[1, 1])


def test_short_data_and_bad_step_rejected():
    m = _msg(RGB, 'rgb8')
    m.data = m.data[:-1]
    with pytest.raises(ValueError):
        image_msg_to_array(m)
    m = _msg(RGB, 'rgb8')
    m.step = 2
    with pytest.raises(ValueError):
        image_msg_to_array(m)


def test_unsupported_encoding_rejected():
    with pytest.raises(ValueError):
        image_msg_to_array(_msg(RGB, 'yuv422'))
    with pytest.raises(ValueError):
        image_msg_to_rgb(_msg(np.zeros((2, 2), np.uint16), '16UC1'))
