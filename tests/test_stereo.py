import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

from vlm_nav.stereo import StereoDepth, baseline_from_projection, box_depth  # noqa: E402


def test_baseline_from_isaac_right_camera_info():
    # values published by the Nova Carter front stereo camera
    P = [958.569051481523, 0.0, 960.0000152587891, -143.78535772, 0.0, 958.569, 600.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert baseline_from_projection(P) == pytest.approx(0.15, abs=1e-6)


def test_baseline_unset():
    assert baseline_from_projection([500.0, 0, 320, 0.0, 0, 500, 240, 0, 0, 0, 1, 0]) is None


def _stereo_pair(w=480, h=320, disparity=24, seed=1):
    rng = np.random.default_rng(seed)
    tex = cv2.GaussianBlur(rng.integers(0, 255, (h, w + 64), dtype=np.uint8), (3, 3), 0)
    left = tex[:, 32:32 + w]
    right = tex[:, 32 + disparity:32 + disparity + w]   # a feature appears `disparity` px further left in the right image
    return np.repeat(left[:, :, None], 3, 2).copy(), np.repeat(right[:, :, None], 3, 2).copy()


def test_stereo_recovers_known_depth():
    fx, baseline, disparity = 500.0, 0.15, 24
    left, right = _stereo_pair(disparity=disparity)
    depth = StereoDepth(fx, baseline, scale=1.0, num_disparities=64).compute(left, right)
    expected = fx * baseline / disparity                                  # 3.125 m
    d = box_depth(depth, (150, 100, 350, 250))
    assert d == pytest.approx(expected, rel=0.03)


def test_stereo_scale_factor_keeps_metric_depth():
    fx, baseline, disparity = 500.0, 0.15, 24
    left, right = _stereo_pair(disparity=disparity)
    depth = StereoDepth(fx, baseline, scale=0.5, num_disparities=64).compute(left, right)
    assert depth.shape == left.shape[:2]
    assert box_depth(depth, (150, 100, 350, 250)) == pytest.approx(fx * baseline / disparity, rel=0.05)


def test_box_depth_uses_centre_not_background_edges():
    depth = np.full((100, 100), 10.0, np.float32)
    depth[30:70, 30:70] = 2.0                        # object, with far background around it
    assert box_depth(depth, (20, 20, 80, 80)) == 2.0


def test_box_depth_rejects_boxes_without_valid_pixels():
    depth = np.full((50, 50), np.nan, np.float32)
    assert box_depth(depth, (10, 10, 40, 40)) is None
    assert box_depth(np.ones((50, 50), np.float32), (60, 60, 70, 70)) is None   # outside the image


def test_invalid_stereo_params():
    with pytest.raises(ValueError):
        StereoDepth(500.0, 0.0)
