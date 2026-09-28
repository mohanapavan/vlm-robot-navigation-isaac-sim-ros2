"""Occupancy-grid helper (pure numpy / PIL, no ROS): the saved Nav2 map, or a live nav_msgs/OccupancyGrid.

Used to check that a goal or a detected object is somewhere the robot can actually be, without asking the planner.
World coordinates are metres in the map frame; row 0 of the image is the *top* (largest y) as in map_server.
"""
import math
import os

import numpy as np

UNKNOWN, FREE, OCCUPIED = -1, 0, 100

# map_saver writes unexplored space as the grey value 205, which map_server reads back as occupancy (255-205)/255 = 0.196.
# map_saver also writes `free_thresh: 0.25`, and 0.196 < 0.25 means *free*: served like that, every unexplored cell is
# open floor (the saved hospital map went from 67 % unknown to 0 %). Reading it with a free threshold below 0.196 keeps
# unexplored space unknown, which is what the map means.
UNKNOWN_GRAY_OCCUPANCY = 0.196


def effective_free_thresh(free_thresh):
    return min(float(free_thresh), UNKNOWN_GRAY_OCCUPANCY - 0.006)


def corrected_map_yaml(yaml_path, out_dir=None):
    """Path of a map yaml that keeps unexplored space unknown when Nav2's map_server loads it.

    Returns `yaml_path` itself when nothing needs correcting. Otherwise writes a copy (the original is never touched)
    with the same image (absolute path) and a lowered `free_thresh`, into `out_dir` (default: the system temp dir).
    """
    import hashlib
    import tempfile

    import yaml
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    if meta.get('mode', 'trinary') != 'trinary' or float(meta.get('free_thresh', 0.25)) <= effective_free_thresh(0.25) + 1e-9:
        return str(yaml_path)
    image = meta['image']
    if not os.path.isabs(image):
        image = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), image)
    meta['image'] = image
    meta['free_thresh'] = effective_free_thresh(meta.get('free_thresh', 0.25))
    out_dir = out_dir or tempfile.gettempdir()
    name = 'vlm_nav_map_' + hashlib.sha1(os.path.abspath(yaml_path).encode()).hexdigest()[:10] + '.yaml'
    out = os.path.join(out_dir, name)
    with open(out, 'w') as f:
        yaml.safe_dump(meta, f)
    return out


class MapGrid:
    def __init__(self, cells, resolution, origin_xy):
        """`cells`: int array (H, W) with -1 unknown / 0 free / 100 occupied, row 0 = lowest y (OccupancyGrid order)."""
        cells = np.asarray(cells)
        if cells.ndim != 2 or cells.size == 0:
            raise ValueError('map must be a non-empty 2-D array')
        if resolution <= 0:
            raise ValueError('map resolution must be positive')
        self.cells = cells.astype(np.int16)
        self.resolution = float(resolution)
        self.origin = (float(origin_xy[0]), float(origin_xy[1]))
        self.height, self.width = self.cells.shape
        self._dist_cache = None

    # ------------------------------------------------------------------ constructors

    @classmethod
    def from_yaml(cls, yaml_path):
        """Load a nav2_map_server map (`my_map.yaml` + its image), honouring negate / thresholds / trinary mode."""
        import yaml
        from PIL import Image
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        image_path = meta['image']
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), image_path)
        img = np.array(Image.open(image_path).convert('L'), dtype=np.float64)
        occ = img / 255.0 if meta.get('negate', 0) else (255.0 - img) / 255.0
        occupied_t, free_t = float(meta.get('occupied_thresh', 0.65)), float(meta.get('free_thresh', 0.25))
        if meta.get('mode', 'trinary') == 'trinary':
            free_t = effective_free_thresh(free_t)          # keep unexplored (grey) pixels unknown, see above
        cells = np.full(img.shape, UNKNOWN, dtype=np.int16)
        cells[occ < free_t] = FREE
        cells[occ > occupied_t] = OCCUPIED
        origin = meta.get('origin', [0.0, 0.0, 0.0])
        return cls(cells[::-1], float(meta['resolution']), (origin[0], origin[1]))

    @classmethod
    def from_occupancy_grid(cls, msg, occupied_threshold=65):
        """From a nav_msgs/OccupancyGrid (values -1 or 0..100)."""
        data = np.array(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        cells = np.where(data < 0, UNKNOWN, np.where(data >= occupied_threshold, OCCUPIED, FREE)).astype(np.int16)
        return cls(cells, msg.info.resolution, (msg.info.origin.position.x, msg.info.origin.position.y))

    # ------------------------------------------------------------------ queries

    def to_cell(self, x, y):
        """(row, col) of a world point, or None when it is outside the map."""
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        col = int(math.floor((x - self.origin[0]) / self.resolution))
        row = int(math.floor((y - self.origin[1]) / self.resolution))
        if 0 <= row < self.height and 0 <= col < self.width:
            return row, col
        return None

    def to_world(self, row, col):
        """Centre of a cell."""
        return (self.origin[0] + (col + 0.5) * self.resolution, self.origin[1] + (row + 0.5) * self.resolution)

    def state(self, x, y):
        """UNKNOWN (also for points outside the map), FREE or OCCUPIED."""
        cell = self.to_cell(x, y)
        return UNKNOWN if cell is None else int(self.cells[cell])

    @property
    def _distance(self):
        """Metres to the nearest occupied cell, for every cell (computed once)."""
        if self._dist_cache is None:
            import cv2
            not_occupied = (self.cells != OCCUPIED).astype(np.uint8)
            if not_occupied.all():
                self._dist_cache = np.full(self.cells.shape, 1e3, dtype=np.float32)
            else:
                self._dist_cache = cv2.distanceTransform(not_occupied, cv2.DIST_L2, 5) * self.resolution
        return self._dist_cache

    def clearance(self, x, y):
        """Distance (m) from a point to the nearest occupied cell; 0.0 outside the map."""
        cell = self.to_cell(x, y)
        return 0.0 if cell is None else float(self._distance[cell])

    def is_free(self, x, y, clearance=0.0):
        """True if the point is in known free space at least `clearance` metres from any occupied cell."""
        cell = self.to_cell(x, y)
        return cell is not None and self.cells[cell] == FREE and self._distance[cell] >= clearance

    def segment_is_free(self, a, b, clearance=0.0, ignore_end=0.0):
        """True if every point on the segment a->b is free (see is_free), except the last `ignore_end` metres."""
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(math.ceil(length / (self.resolution * 0.5))))
        for i in range(n + 1):
            t = i / n
            if length * (1 - t) < ignore_end:
                break
            if not self.is_free(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, clearance):
                return False
        return True

    def last_free_point(self, a, b, clearance=0.0, near=0.6):
        """Walking from `a` toward `b`, the last point before the first blocked one (None if blocked straight away).

        Clearance is not demanded within `near` metres of `a`, where the robot may already be close to a wall.
        """
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(math.ceil(length / self.resolution)))
        last = None
        for i in range(1, n + 1):
            t = i / n
            p = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            if not self.is_free(p[0], p[1], 0.0 if length * t < near else clearance):
                break
            last = p
        return last

    def free_fraction(self):
        known = self.cells != UNKNOWN
        return float((self.cells == FREE).sum() / max(1, known.sum()))

    def nearest_free(self, x, y, clearance=0.0, max_radius=2.0):
        """Closest free point (with clearance) to (x, y) within `max_radius` metres, or None."""
        if self.is_free(x, y, clearance):
            return x, y
        step = self.resolution * 2
        best = None
        r = step
        while r <= max_radius:
            for k in range(max(8, int(2 * math.pi * r / step))):
                ang = 2 * math.pi * k / max(8, int(2 * math.pi * r / step))
                px, py = x + r * math.cos(ang), y + r * math.sin(ang)
                if self.is_free(px, py, clearance):
                    best = (px, py)
                    break
            if best:
                return best
            r += step
        return None
