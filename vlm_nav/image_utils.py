"""Decode sensor_msgs/Image into numpy without cv_bridge, honouring `encoding`, `step` and endianness."""
import numpy as np

# encoding -> (channels, numpy dtype)
_ENCODINGS = {
    'rgb8': (3, np.uint8), 'bgr8': (3, np.uint8),
    'rgba8': (4, np.uint8), 'bgra8': (4, np.uint8),
    'mono8': (1, np.uint8), '8UC1': (1, np.uint8),
    '8UC3': (3, np.uint8), '8UC4': (4, np.uint8),
    'mono16': (1, np.uint16), '16UC1': (1, np.uint16),
    '32FC1': (1, np.float32),
}


def image_msg_to_array(msg):
    """Return the image as an array shaped (H, W, C), or (H, W) for single-channel encodings.

    Works on anything with height/width/step/encoding/is_bigendian/data, i.e. a
    sensor_msgs/Image. Rows are read using `step`, so padded rows are handled.
    """
    try:
        channels, dtype = _ENCODINGS[msg.encoding]
    except KeyError:
        raise ValueError(f'unsupported image encoding {msg.encoding!r}') from None
    dtype = np.dtype(dtype)
    if dtype.itemsize > 1:
        dtype = dtype.newbyteorder('>' if msg.is_bigendian else '<')
    bytes_per_row = msg.width * channels * dtype.itemsize
    step = msg.step or bytes_per_row
    if step < bytes_per_row:
        raise ValueError(f'image step {step} smaller than a row ({bytes_per_row} bytes)')
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if buf.size < step * msg.height:
        raise ValueError(f'image data too short: {buf.size} < {step * msg.height}')
    rows = buf[:step * msg.height].reshape(msg.height, step)[:, :bytes_per_row]
    arr = np.ascontiguousarray(rows).view(dtype).reshape(msg.height, msg.width, channels)
    arr = arr.astype(dtype.newbyteorder('='), copy=False)
    return arr[:, :, 0] if channels == 1 else arr


def image_msg_to_rgb(msg):
    """Return an (H, W, 3) uint8 RGB array for any supported 8-bit colour or mono encoding."""
    arr = image_msg_to_array(msg)
    if arr.dtype != np.uint8:
        raise ValueError(f'cannot convert {msg.encoding!r} to 8-bit RGB')
    if arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2)
    if msg.encoding in ('bgr8', 'bgra8'):
        arr = arr[:, :, [2, 1, 0]]
    return np.ascontiguousarray(arr[:, :, :3])
