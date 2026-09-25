"""Tests for spsem.schabenberger_spatial_error_model.

Reference value is spatialreg 1.3 errorsarlm(y ~ x1 + x2, style "B",
method "eigen"): logLik -25.123642333442028.
"""

import math

import pytest

NR, NC = 5, 4
N = NR * NC
# rook adjacency on a 5 x 4 grid, sites numbered row by row
W = [[1.0 if abs(i // NC - j // NC) + abs(i % NC - j % NC) == 1 else 0.0
      for j in range(N)] for i in range(N)]
Z = [((i * 37) % 17) / 3 + 0.5 * (i // NC) for i in range(N)]
X1 = [float((i * 5) % 9) for i in range(N)]
X2 = [math.cos(i) for i in range(N)]
Y = [1 + 0.8 * a - 0.5 * b + ((i * 13) % 7 - 3) / 2 + 0.3 * (i // NC)
     for i, (a, b) in enumerate(zip(X1, X2))]
X = [[1.0, a, b] for a, b in zip(X1, X2)]

from morie.fn.spsem import _neg2ll, schabenberger_spatial_error_model


def test_spsem_basic():
    """The maximised likelihood matches errorsarlm."""
    r = schabenberger_spatial_error_model(X, Y, W)
    assert r["neg2loglik"] == pytest.approx(-2.0 * -25.123642333442028, rel=1e-10)
    assert r["n"] == N and r["k"] == 3
    lo, hi = r["rho_bounds"]
    assert lo < r["rho"] < hi


def test_spsem_edge():
    """rho is a profile optimum and beta is GLS on the whitened data at that rho."""
    r = schabenberger_spatial_error_model(X, Y, W)
    rho = r["rho"]
    f0 = r["neg2loglik"]
    for h in (1e-3, 1e-4):
        assert _neg2ll(Y, X, W, rho - h)[0] >= f0
        assert _neg2ll(Y, X, W, rho + h)[0] >= f0
    a = [[(1.0 if i == j else 0.0) - rho * W[i][j] for j in range(N)] for i in range(N)]
    ys = [sum(a[i][j] * Y[j] for j in range(N)) for i in range(N)]
    xs = [[sum(a[i][j] * X[j][c] for j in range(N)) for c in range(3)] for i in range(N)]
    from morie.fn._spx import solve
    g = [[sum(xs[i][p] * xs[i][q] for i in range(N)) for q in range(3)] for p in range(3)]
    b = solve(g, [sum(xs[i][p] * ys[i] for i in range(N)) for p in range(3)])
    for got, ref in zip(r["beta"], b):
        assert got == pytest.approx(ref, rel=1e-10)
