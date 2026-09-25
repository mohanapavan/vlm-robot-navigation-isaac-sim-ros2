"""Pure-numpy geometry helpers (no ROS / torch imports so they are easy to unit-test)."""
import math

import numpy as np


def quat_to_yaw(x, y, z, w):
    """Yaw (rotation about +Z) of a quaternion, in radians."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_to_quat(yaw):
    """Quaternion (x, y, z, w) for a pure rotation of `yaw` radians about +Z."""
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def normalize_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def rotate_relative_offset(x, y, yaw, dx, dy):
    """Turn a robot-frame offset (dx forward, dy left) into a point in the frame the pose is in."""
    c, s = math.cos(yaw), math.sin(yaw)
    return x + dx * c - dy * s, y + dx * s + dy * c


def yaw_facing(from_xy, to_xy):
    """Heading (radians) that points from `from_xy` toward `to_xy`."""
    return math.atan2(to_xy[1] - from_xy[1], to_xy[0] - from_xy[0])


def standoff_point(obj_xy, robot_xy, standoff):
    """Point `standoff` metres from the object on the line toward the robot.

    If the robot is already inside the stand-off radius, the robot's own position
    is returned so it only turns to face the object instead of backing away.
    """
    dx = robot_xy[0] - obj_xy[0]
    dy = robot_xy[1] - obj_xy[1]
    dist = math.hypot(dx, dy)
    if dist <= standoff:
        return robot_xy[0], robot_xy[1]
    r = standoff / dist
    return obj_xy[0] + dx * r, obj_xy[1] + dy * r


def transform_to_matrix(translation, quaternion_xyzw):
    """4x4 homogeneous matrix from a translation (x, y, z) and quaternion (x, y, z, w)."""
    x, y, z, w = quaternion_xyzw
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        raise ValueError('zero-length quaternion')
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    m = np.eye(4)
    m[:3, :3] = [[1.0 - (yy + zz), xy - wz, xz + wy],
                 [xy + wz, 1.0 - (xx + zz), yz - wx],
                 [xz - wy, yz + wx, 1.0 - (xx + yy)]]
    m[:3, 3] = translation
    return m


def transform_point(matrix, point):
    p = np.array([point[0], point[1], point[2], 1.0], dtype=float)
    return (matrix @ p)[:3]


def backproject(u, v, depth, fx, fy, cx, cy):
    """Pixel (u, v) with metric depth -> 3-D point in the camera *optical* frame (x right, y down, z forward)."""
    return np.array([(u - cx) * depth / fx, (v - cy) * depth / fy, depth], dtype=float)
