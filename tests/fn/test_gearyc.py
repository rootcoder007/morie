"""Tests for gearyc.gearyc."""

import numpy as np
import pytest

from morie.fn.gearyc import gearyc


def _path_w(n):
    W = np.zeros((n, n))
    for i in range(n - 1):
        W[i, i + 1] = W[i + 1, i] = 1.0
    return W


def _rook_w(side):
    n = side * side
    W = np.zeros((n, n))
    for r in range(side):
        for c in range(side):
            i = r * side + c
            if r + 1 < side:
                W[i, i + side] = W[i + side, i] = 1.0
            if c + 1 < side:
                W[i, i + 1] = W[i + 1, i] = 1.0
    return W


def test_gearyc_matches_a_hand_computed_value():
    """Path graph 1-2-3 with x = (0, 1, 2): each of the four directed
    neighbour pairs contributes (Z_i - Z_j)^2 = 1, S0 = 4,
    sum (Z_i - Zbar)^2 = 2, so c = (n-1)/(2 S0) * 4/2 = 0.5 exactly
    (Schabenberger & Gotway 2005, eq. 1.15)."""
    r = gearyc([0.0, 1.0, 2.0], _path_w(3))
    assert float(r["value"]) == pytest.approx(0.5, abs=1e-12)


def test_gearyc_null_expectation_is_one_and_sides_are_correct():
    """E[c] = 1 under no autocorrelation; c < 1 for clustering, c > 1 for
    a checkerboard. Direction is the OPPOSITE of Moran's I, which is the
    classic way to catch a sign slip."""
    side = 8
    W = _rook_w(side)
    grad = np.add.outer(np.arange(side), np.arange(side)).ravel().astype(float)
    assert float(gearyc(grad, W)["value"]) < 0.6

    checker = np.indices((side, side)).sum(axis=0).ravel() % 2 * 2.0 - 1.0
    assert float(gearyc(checker, W)["value"]) > 1.4

    # White noise: mean of c over seeds sits near 1. Measured 1.00 +/- 0.03.
    vals = []
    for s in range(10):
        rng = np.random.default_rng(s)
        vals.append(float(gearyc(rng.normal(size=side * side), W)["value"]))
    assert np.mean(vals) == pytest.approx(1.0, abs=0.06)


def test_gearyc_is_location_and_scale_invariant():
    rng = np.random.default_rng(3)
    x = rng.normal(size=25)
    W = _rook_w(5)
    a = float(gearyc(x, W)["value"])
    b = float(gearyc(5.0 + 3.0 * x, W)["value"])
    assert a == pytest.approx(b, rel=1e-12)


def test_gearyc_rejects_bad_input():
    with pytest.raises(ValueError, match="must be"):
        gearyc([1.0, 2.0, 3.0], np.zeros((2, 2)))
    with pytest.raises(ValueError, match="sums to zero"):
        gearyc([1.0, 2.0, 3.0], np.zeros((3, 3)))
    with pytest.raises(ValueError, match="zero variance"):
        gearyc([2.0, 2.0, 2.0], _path_w(3))
