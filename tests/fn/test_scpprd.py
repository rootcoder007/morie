"""Tests for morie.fn.scpprd: fitted means of the spatial-lag count model."""

import math

from morie.fn._qpcore import solve
from morie.fn.scpprd import scpprd, scpprd_fn

W = [
    [0.0, 1.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.5, 0.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 0.0, 0.5],
    [0.0, 0.0, 0.0, 1.0, 0.0],
]
X = [[1.0, 0.2], [1.0, -0.4], [1.0, 0.9], [1.0, 0.3], [1.0, 1.4]]


def test_fitted_recomputed():
    rho, b = 0.45, [0.3, 0.8]
    A = [[(1.0 if i == j else 0.0) - rho * W[i][j] for j in range(5)] for i in range(5)]
    eta = solve(A, [b[0] + b[1] * x[1] for x in X])
    r = scpprd(b, X, W, rho=rho)
    assert max(abs(a - math.exp(e)) for a, e in zip(r.fitted, eta)) < 1e-12
    assert abs(r.total - math.fsum(math.exp(e) for e in eta)) < 1e-12


def test_default_rho_and_alias():
    assert scpprd([0.3, 0.8], X, W).fitted == scpprd([0.3, 0.8], X, W, rho=0.2).fitted
    assert scpprd_fn is scpprd
