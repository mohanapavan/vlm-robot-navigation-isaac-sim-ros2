import math

import numpy as np
import pytest

from vlm_nav.geometry import (backproject, normalize_angle, quat_to_yaw, rotate_relative_offset,
                              standoff_point, transform_point, transform_to_matrix, yaw_facing,
                              yaw_to_quat)


@pytest.mark.parametrize('yaw', [0.0, 0.5, math.pi / 2, -2.0, 3.0])
def test_yaw_quaternion_round_trip(yaw):
    assert quat_to_yaw(*yaw_to_quat(yaw)) == pytest.approx(yaw)


def test_relative_move_follows_heading():
    # Robot facing +y (yaw 90 deg): "forward 1 m" must go to +y, "left 1 m" must go to -x.
    x, y = rotate_relative_offset(2.0, 3.0, math.pi / 2, 1.0, 0.0)
    assert (x, y) == pytest.approx((2.0, 4.0))
    x, y = rotate_relative_offset(2.0, 3.0, math.pi / 2, 0.0, 1.0)
    assert (x, y) == pytest.approx((1.0, 3.0))


def test_relative_move_identity_heading():
    assert rotate_relative_offset(1.0, 1.0, 0.0, 2.0, -1.0) == pytest.approx((3.0, 0.0))


def test_standoff_stops_short_of_object_on_robot_side():
    gx, gy = standoff_point((10.0, 0.0), (0.0, 0.0), 0.8)
    assert (gx, gy) == pytest.approx((9.2, 0.0))
    assert math.hypot(gx - 10.0, gy) == pytest.approx(0.8)


def test_standoff_does_not_back_away_when_already_close():
    assert standoff_point((1.0, 0.0), (0.5, 0.0), 0.8) == (0.5, 0.0)


def test_goal_yaw_faces_object():
    assert yaw_facing((0.0, 0.0), (0.0, 5.0)) == pytest.approx(math.pi / 2)
    assert yaw_facing((1.0, 1.0), (0.0, 1.0)) == pytest.approx(math.pi)


def test_normalize_angle():
    assert normalize_angle(3 * math.pi) == pytest.approx(math.pi, abs=1e-9) or \
        normalize_angle(3 * math.pi) == pytest.approx(-math.pi, abs=1e-9)


def test_optical_frame_point_lands_in_front_of_robot():
    # front camera as published by Isaac Sim: optical rotation, mounted 0.1 m ahead of base_link.
    m = transform_to_matrix((0.1, 0.075, 0.346), (0.5, -0.5, 0.5, -0.5))
    centre = transform_point(m, backproject(960, 600, 3.0, 958.0, 958.0, 960.0, 600.0))
    assert centre == pytest.approx([3.1, 0.075, 0.346], abs=1e-6)
    # a pixel to the right of the image centre is to the robot's right (-y)
    right = transform_point(m, backproject(960 + 958, 600, 3.0, 958.0, 958.0, 960.0, 600.0))
    assert right[1] == pytest.approx(0.075 - 3.0, abs=1e-6)
    # a pixel below the centre is lower (-z)
    below = transform_point(m, backproject(960, 600 + 958, 3.0, 958.0, 958.0, 960.0, 600.0))
    assert below[2] == pytest.approx(0.346 - 3.0, abs=1e-6)


def test_transform_composes_with_yaw():
    m = transform_to_matrix((1.0, 2.0, 0.0), yaw_to_quat(math.pi / 2))
    assert transform_point(m, (1.0, 0.0, 0.0)) == pytest.approx(np.array([1.0, 3.0, 0.0]))


def test_zero_quaternion_rejected():
    with pytest.raises(ValueError):
        transform_to_matrix((0, 0, 0), (0, 0, 0, 0))
