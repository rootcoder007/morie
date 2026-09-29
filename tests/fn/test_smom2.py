"""Tests for morie.fn.smom2: recompute from the definition."""

import math

from morie.fn.smom2 import central_moment

X = [2.5, -1.0, 4.25, 0.5, 3.0, -2.75, 1.5, 6.0]


def test_central_moments():
    m = sum(X) / 8
    for k in (2, 3, 4):
        assert abs(central_moment(X, k=k).value - math.fsum((v - m) ** k for v in X) / 8) < 1e-12
    assert abs(central_moment(X, k=1).value) < 1e-15
