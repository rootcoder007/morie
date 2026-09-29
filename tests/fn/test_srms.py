"""Tests for morie.fn.srms: recompute from the definition."""

import math

from morie.fn.srms import rms_value

X = [2.5, -1.0, 4.25, 0.5, 3.0, -2.75, 1.5, 6.0]


def test_rms():
    assert abs(rms_value(X).value - math.sqrt(math.fsum(v * v for v in X) / 8)) < 1e-15
    assert rms_value([3.0, -4.0]).value == math.sqrt(12.5)
