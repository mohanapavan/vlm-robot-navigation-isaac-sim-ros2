import math
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from vlm_nav.map_grid import FREE, OCCUPIED, UNKNOWN, MapGrid


def _grid():
    """10 m x 6 m at 0.1 m/cell, origin (-5, -3): free floor, a wall at x = 2, an unknown strip for y > 2."""
    cells = np.zeros((60, 100), dtype=np.int16)
    cells[:, 70:73] = OCCUPIED            # wall x in [2.0, 2.3)
    cells[50:, :] = UNKNOWN               # y >= 2.0
    return MapGrid(cells, 0.1, (-5.0, -3.0))


def test_states_and_bounds():
    g = _grid()
    assert g.state(0.0, 0.0) == FREE
    assert g.state(2.1, 0.0) == OCCUPIED
    assert g.state(0.0, 2.5) == UNKNOWN
    assert g.state(50.0, 0.0) == UNKNOWN                # outside the map counts as unknown
    assert g.state(float('nan'), 0.0) == UNKNOWN
    assert g.to_cell(-5.0, -3.0) == (0, 0) and g.to_cell(4.99, 2.99) == (59, 99)


def test_clearance_and_is_free():
    g = _grid()
    assert g.clearance(1.0, 0.0) == pytest.approx(1.0, abs=0.1)
    assert g.is_free(0.0, 0.0, clearance=0.5) and not g.is_free(1.8, 0.0, clearance=0.5)
    assert not g.is_free(2.1, 0.0) and not g.is_free(0.0, 2.5) and not g.is_free(99, 99)


def test_segment_and_last_free_point():
    g = _grid()
    assert g.segment_is_free((-3, 0), (1, 0))
    assert not g.segment_is_free((0, 0), (3, 0))                        # crosses the wall
    assert g.segment_is_free((0, 0), (2.1, 0), ignore_end=0.5)          # only the last bit touches the wall
    stop = g.last_free_point((0.0, 0.0), (4.0, 0.0), clearance=0.4)
    assert stop is not None and 1.4 < stop[0] < 1.7                     # stops short of the wall, with clearance
    assert g.last_free_point((1.95, 0.0), (4.0, 0.0)) is None           # already against the wall


def test_nearest_free():
    g = _grid()
    x, y = g.nearest_free(2.1, 0.0, clearance=0.3, max_radius=1.5)
    assert g.is_free(x, y, 0.3) and math.hypot(x - 2.1, y) < 1.0
    assert g.nearest_free(0.0, 2.5, max_radius=0.2) is None             # nothing free within reach


def test_from_yaml_honours_row_order_and_thresholds(tmp_path):
    img = np.full((4, 6), 254, dtype=np.uint8)      # 254 free
    img[0, 0] = 0                                   # top-left occupied -> highest y, lowest x
    img[3, 5] = 205                                 # bottom-right unknown -> lowest y, highest x
    Image.fromarray(img).save(tmp_path / 'm.pgm')
    (tmp_path / 'm.yaml').write_text('image: m.pgm\nresolution: 0.5\norigin: [10.0, 20.0, 0]\nnegate: 0\n'
                                     'occupied_thresh: 0.65\nfree_thresh: 0.25\n')
    g = MapGrid.from_yaml(tmp_path / 'm.yaml')
    assert g.state(10.25, 21.75) == OCCUPIED        # x 10.0-10.5, y 21.5-22.0 (top row)
    assert g.state(12.75, 20.25) == UNKNOWN         # bottom-right
    assert g.state(11.0, 21.0) == FREE
    assert g.width == 6 and g.height == 4 and g.origin == (10.0, 20.0)


def test_from_occupancy_grid_message():
    msg = SimpleNamespace(data=[0, 100, -1, 50], info=SimpleNamespace(
        height=2, width=2, resolution=1.0, origin=SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0))))
    g = MapGrid.from_occupancy_grid(msg)
    assert (g.state(0.5, 0.5), g.state(1.5, 0.5), g.state(0.5, 1.5), g.state(1.5, 1.5)) == (FREE, OCCUPIED, UNKNOWN, FREE)


def test_bad_maps_are_rejected():
    with pytest.raises(ValueError):
        MapGrid(np.zeros((0, 0)), 0.05, (0, 0))
    with pytest.raises(ValueError):
        MapGrid(np.zeros((2, 2)), 0.0, (0, 0))
