"""Tests for theteap2.theta_map (MAP ability, Mislevy 1986)."""

import math

import pytest


ITEMS = [[1.2, -0.8, 0.0], [0.9, -0.2, 0.1], [1.5, 0.3, 0.2], [0.7, 1.0, 0.0], [1.1, 1.6, 0.15]]
X = [[1, 1, 0, 1, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 0, 1, 0, 0]]


def _logpost(t, y, mu=0.0, sd=1.0):
    lp = -(t - mu) ** 2 / (2 * sd * sd)
    for (a, b, c), r in zip(ITEMS, y):
        p = c + (1 - c) / (1 + math.exp(-a * (t - b)))
        lp += math.log(p) if r else math.log(1 - p)
    return lp

from morie.fn.theteap2 import theta_map


def _mode(y):
    """Posterior mode by bisection on a central-difference score."""
    def d(t):
        return (_logpost(t + 1e-6, y) - _logpost(t - 1e-6, y)) / 2e-6
    lo, hi = -6.0, 6.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if d(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)


def test_theteap2_basic():
    """The mode solves the posterior score equation (bisection on a
    numerical score, accurate to ~1e-9 given the 1e-6 difference step);
    se = 1/sqrt(I(theta) + 1/sigma^2) with the 3PL Fisher information."""
    r = theta_map(X, ITEMS)
    for i, y in enumerate(X):
        t = float(r["theta"][i])
        assert t == pytest.approx(_mode(y), abs=1e-7)
        info = 0.0
        for (a, b, c) in ITEMS:
            ps = 1 / (1 + math.exp(-a * (t - b)))
            p = c + (1 - c) * ps
            dp = a * (1 - c) * ps * (1 - ps)
            info += dp * dp / (p * (1 - p))
        assert float(r["se"][i]) == pytest.approx(1 / math.sqrt(info + 1.0), rel=1e-9)


def test_theteap2_edge():
    """MAP exists for the perfect patterns (the prior bounds it); the
    item table must match the response columns."""
    r = theta_map([X[1], X[2]], ITEMS)
    assert all(math.isfinite(float(v)) for v in r["theta"])
    with pytest.raises(ValueError):
        theta_map(X, ITEMS[:3])
