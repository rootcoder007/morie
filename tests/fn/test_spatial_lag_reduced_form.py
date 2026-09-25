"""Tests for spatial_lag_reduced_form (SAR reduced form, eq. 12.4)."""

import math

import pytest

from morie.fn.spatial_lag_reduced_form import spatial_lag_reduced_form


def _w(n=6):
    A = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(n)] for i in range(n)]
    return [[v / sum(r) for v in r] for r in A]


def test_spatial_lag_reduced_form_basic():
    """y solves y = rho W y + xb + e exactly, and for |rho| < 1 with a
    row-standardised W equals the Neumann series sum_k rho^k W^k (xb+e)."""
    W = _w()
    xb = [1.0, 2.0, 0.5, -1.0, 0.0, 3.0]
    e = [0.1 * math.sin(k) for k in range(6)]
    rho = 0.4
    r = spatial_lag_reduced_form(rho, W, xb, e)
    y = [float(v) for v in r["y"]]
    assert r["value"] == y[0]
    for i in range(6):
        assert y[i] == pytest.approx(rho * sum(W[i][j] * y[j] for j in range(6)) + xb[i] + e[i], abs=1e-12)
    v = [a + b for a, b in zip(xb, e)]
    s, term = v[:], v[:]
    for _ in range(200):
        term = [rho * sum(W[i][j] * term[j] for j in range(6)) for i in range(6)]
        s = [a + b for a, b in zip(s, term)]
    assert y == pytest.approx(s, abs=1e-12)


def test_spatial_lag_reduced_form_edge():
    """rho = 0 returns xb + e; a singular I - rho W and bad shapes raise."""
    W = _w()
    xb = [1.0] * 6
    e = [0.5] * 6
    r = spatial_lag_reduced_form(0.0, W, xb, e)
    y = [float(v) for v in r["y"]]
    assert r["value"] == y[0]
    assert y == pytest.approx([1.5] * 6, abs=1e-15)
    with pytest.raises(ValueError):
        spatial_lag_reduced_form(1.0, W, xb, e)
    with pytest.raises(ValueError):
        spatial_lag_reduced_form(0.3, W, xb[:-1], e)
