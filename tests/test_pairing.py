from types import SimpleNamespace

from vlm_nav.pairing import StereoPairer, stamp_ns

MS = 1_000_000


def test_stamp_ns():
    assert stamp_ns(SimpleNamespace(stamp=SimpleNamespace(sec=41, nanosec=366_666_666))) == 41_366_666_666


def test_pairs_regardless_of_arrival_order():
    p = StereoPairer(5 * MS)
    assert p.add_left(100 * MS, 'L1') is None
    assert p.add_right(102 * MS, 'R1') == ('L1', 'R1')          # right arrives after left
    assert p.add_right(200 * MS, 'R2') is None
    assert p.add_left(199 * MS, 'L2') == ('L2', 'R2')           # right arrives before left


def test_picks_nearest_partner_within_tolerance_only():
    p = StereoPairer(5 * MS)
    p.add_right(100 * MS, 'Ra')
    p.add_right(104 * MS, 'Rb')
    assert p.add_left(103 * MS, 'L') == ('L', 'Rb')
    p2 = StereoPairer(5 * MS)
    p2.add_right(100 * MS, 'R')
    assert p2.add_left(110 * MS, 'L') is None                   # 10 ms apart: a different frame


def test_dropped_frames_are_counted_not_mispaired():
    p = StereoPairer(5 * MS, window=3)
    for i in range(6):                                          # only left frames arrive
        assert p.add_left(i * 33 * MS, f'L{i}') is None
    assert p.unmatched == 3
    assert p.add_right(5 * 33 * MS, 'R5') == ('L5', 'R5')       # newest left is still waiting
