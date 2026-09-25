"""Tests for bsacorr.rangayyan_ch4_ccf_continuous_with_delay."""

import pytest

from morie.fn.bsacorr import rangayyan_ch4_ccf_continuous_with_delay as xcc

T = [0.1 * i for i in range(11)]
X = [0.0, 0.5, 1.0, 0.7, 0.2, -0.3, -0.6, -0.2, 0.1, 0.4, 0.0]
Y = [1.0, 0.8, 0.1, -0.4, -0.2, 0.3, 0.9, 0.5, 0.0, -0.1, 0.2]


def _trap(pts):
    return sum(0.5 * (pts[i][1] + pts[i + 1][1]) * (pts[i + 1][0] - pts[i][0])
               for i in range(len(pts) - 1))


def test_rng201_basic():
    """R_xy(tau) = int x(t) y(t + tau) dt by the trapezoid rule over the
    overlap: a delay of two samples pairs x_i with y_{i+2}; half a
    sample uses the linear midpoint of y, both recomputed."""
    r = xcc(X, Y, T, [0.2, 0.05])
    two = _trap([(T[i], X[i] * Y[i + 2]) for i in range(9)])
    half = _trap([(T[i], X[i] * 0.5 * (Y[i] + Y[i + 1])) for i in range(10)])
    assert r["ccf"] == pytest.approx([two, half], rel=1e-12)


def test_rng201_edge():
    """A scalar delay returns a scalar; mismatched lengths are refused."""
    r = xcc(X, Y, T, 0.0)
    assert r["ccf"] == pytest.approx(_trap([(t, a * b) for t, a, b in zip(T, X, Y)]), rel=1e-14)
    with pytest.raises(ValueError):
        xcc(X[:-1], Y, T, 0.1)
