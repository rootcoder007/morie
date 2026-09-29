"""Tests for morie.fn.qrsdr."""

import math

from morie.fn.qrsdr import qrs_duration


def test_durations():
    on, off = [100, 460, 830, 1200], [125, 482, 858, 1224]
    r = qrs_duration(on, off, fs=250.0)
    d = [(b - a) / 250 for a, b in zip(on, off)]
    m = sum(d) / 4
    assert abs(r.value - m) < 1e-15
    assert abs(r.extra["std_dur"] - math.sqrt(sum((t - m) ** 2 for t in d) / 3)) < 1e-15
    assert qrs_duration([], []).value == 0.0
