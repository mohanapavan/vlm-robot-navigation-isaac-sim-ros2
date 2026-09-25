"""Metric depth from a rectified stereo pair (the robot has cameras but no depth sensor)."""
import cv2
import numpy as np


def baseline_from_projection(P):
    """Stereo baseline (m) from the *right* camera_info projection matrix P (row-major, 12 values).

    P[3] = -fx * baseline for a right camera. Returns None when it isn't set
    (some simulators publish 0), in which case the caller must supply a baseline.
    """
    fx, tx = P[0], P[3]
    if fx > 0 and tx < 0:
        return -tx / fx
    return None


class StereoDepth:
    """Semi-global block matching wrapper returning a metric depth image (NaN where invalid)."""

    def __init__(self, fx, baseline, scale=0.5, num_disparities=128, block_size=5,
                 min_depth=0.3, max_depth=15.0):
        if baseline <= 0 or fx <= 0:
            raise ValueError('fx and baseline must be positive')
        self.fx, self.baseline, self.scale = fx, baseline, scale
        self.min_depth, self.max_depth = min_depth, max_depth
        num_disparities = max(16, int(round(num_disparities / 16.0)) * 16)
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=num_disparities, blockSize=block_size,
            P1=8 * block_size ** 2, P2=32 * block_size ** 2,
            disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    def compute(self, left_rgb, right_rgb):
        """Depth in metres, same size as the input, float32; NaN = no valid match."""
        h, w = left_rgb.shape[:2]
        left = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2GRAY)
        right = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2GRAY)
        if self.scale != 1.0:
            size = (max(1, int(w * self.scale)), max(1, int(h * self.scale)))
            left = cv2.resize(left, size, interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, size, interpolation=cv2.INTER_AREA)
        disp = self.matcher.compute(left, right).astype(np.float32) / 16.0
        with np.errstate(divide='ignore', invalid='ignore'):
            depth = (self.fx * self.scale * self.baseline) / disp
        depth[(disp <= 0.5) | (depth < self.min_depth) | (depth > self.max_depth)] = np.nan
        if self.scale != 1.0:
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        return depth


def box_depth(depth, box_xyxy, inner_fraction=0.5, min_valid_fraction=0.1):
    """Median depth over the central part of a detection box, or None if too few valid pixels.

    Using the box centre region (not the whole box) keeps background pixels at
    the edges of the box from dragging the estimate behind the object.
    """
    h, w = depth.shape
    x0, y0, x1, y1 = box_xyxy
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    hw, hh = (x1 - x0) * inner_fraction / 2.0, (y1 - y0) * inner_fraction / 2.0
    xa, xb = int(max(0, np.floor(cx - hw))), int(min(w, np.ceil(cx + hw)))
    ya, yb = int(max(0, np.floor(cy - hh))), int(min(h, np.ceil(cy + hh)))
    if xb <= xa or yb <= ya:
        return None
    patch = depth[ya:yb, xa:xb]
    valid = patch[np.isfinite(patch)]
    if valid.size < max(1, min_valid_fraction * patch.size):
        return None
    return float(np.median(valid))
