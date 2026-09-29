"""Tests for morie.fn.smom1: recompute from the definition."""

import math

from morie.fn.smom1 import raw_moment

X = [2.5, -1.0, 4.25, 0.5, 3.0, -2.75, 1.5, 6.0]


def test_raw_moments():
    for k in (1, 2, 3, 4):
        assert abs(raw_moment(X, k=k).value - math.fsum(v**k for v in X) / 8) < 1e-12
