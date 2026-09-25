"""Pair left/right stereo images by header stamp (pure python, no ROS imports)."""


def stamp_ns(header):
    """Header stamp as integer nanoseconds."""
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


class StereoPairer:
    """Pair left/right images by header stamp, whichever order they arrive in.

    Best-effort camera streams drop frames unevenly, so each side keeps a small window of frames
    waiting for a partner within `tolerance_ns`; frames that fall out of the window unpaired are counted.
    """

    def __init__(self, tolerance_ns, window=8):
        self.tolerance_ns = tolerance_ns
        self.window = window
        self._left, self._right = [], []
        self.unmatched = 0

    def _take_best(self, buf, stamp):
        best = None
        for i, (s, _) in enumerate(buf):
            d = abs(s - stamp)
            if d <= self.tolerance_ns and (best is None or d < best[0]):
                best = (d, i)
        return buf.pop(best[1])[1] if best else None

    def _push(self, buf, item):
        buf.append(item)
        if len(buf) > self.window:
            buf.pop(0)
            self.unmatched += 1

    def add_left(self, stamp, msg):
        """Returns (left, right) once a matching right frame is known, else None."""
        right = self._take_best(self._right, stamp)
        if right is not None:
            return msg, right
        self._push(self._left, (stamp, msg))
        return None

    def add_right(self, stamp, msg):
        """Returns (left, right) once a matching left frame is known, else None."""
        left = self._take_best(self._left, stamp)
        if left is not None:
            return left, msg
        self._push(self._right, (stamp, msg))
        return None
