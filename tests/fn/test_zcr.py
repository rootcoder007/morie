"""Tests for morie.fn.zcr."""

import math

from morie.fn.zcr import zero_crossing_rate


def test_rate_and_frames():
    x = [math.sin(0.9 * n + 0.2) for n in range(60)]
    s = [1 if t >= 0 else -1 for t in x]
    ref = sum(1 for i in range(1, 60) if s[i] != s[i - 1]) / 59
    assert abs(zero_crossing_rate(x).value - ref) < 1e-15
    r = zero_crossing_rate(x, frame_length=20)
    per = [sum(1 for i in range(1, 20) if s[20 * f + i] != s[20 * f + i - 1]) / 19 for f in range(3)]
    assert r.extra["per_frame"] == per
    assert abs(r.value - sum(per) / 3) < 1e-15
