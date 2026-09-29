"""Tests for morie.fn.popvar: recompute from the definition."""

import math

from morie.fn.popvar import popvar

X = [2.5, -1.0, 4.25, 0.5, 3.0, -2.75, 1.5, 6.0]


def test_population_variance():
    m = sum(X) / 8
    r = popvar(X)
    assert abs(r["variance"] - math.fsum((v - m) ** 2 for v in X) / 8) < 1e-14
    assert r["identity_error"] < 1e-12


def test_large_offset_uses_two_pass():
    Y = [1e9 + v for v in X]
    m = math.fsum(X) / 8
    assert abs(popvar(Y)["variance"] - math.fsum((v - m) ** 2 for v in X) / 8) < 1e-6
