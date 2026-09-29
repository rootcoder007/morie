"""Tests for morie.fn.scpflx: the stacked fixed-effects SAR-Poisson model."""

import math

from morie.fn.scpflx import scpflx
from morie.fn.spcount import sar_poisson

W = [[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]]
X = [[0.1], [0.5], [0.3], [0.4], [0.9], [0.2], [0.6], [0.8], [0.7]]
Y = [1, 4, 2, 2, 6, 1, 3, 7, 4]
IDS = ["a", "b", "c"] * 3


def test_equals_sar_poisson_on_dummies_and_block_weights():
    Z = [[X[i][0]] + [1.0 if IDS[i] == u else 0.0 for u in "abc"] for i in range(9)]
    Wf = [[W[i % 3][j % 3] if i // 3 == j // 3 else 0.0 for j in range(9)] for i in range(9)]
    f = sar_poisson(Y, Z, Wf)
    r = scpflx(Y, X, W, IDS)
    assert abs(r.statistic - f.rho) < 1e-6
    assert abs(r.extra["beta"][0] - f.coefficients[0]) < 1e-6
    assert abs(r.extra["unit_effects"]["c"] - f.coefficients[3]) < 1e-6
    assert r.extra["periods"] == 3


def test_bad_shapes_rejected():
    import pytest

    with pytest.raises(ValueError):
        scpflx(Y[:8], X[:8], W, IDS[:8])


def test_profile_score_vanishes_at_rho():
    # (y - mu)' (I - rho W)^{-1} W eta = 0 at the maximum, eta = log mu (block-diagonal W)
    r = scpflx(Y, X, W, IDS)
    rho, mu = r.statistic, r.extra["fitted"]
    eta = [math.log(m) for m in mu]
    s = 0.0
    for t in range(3):
        blk = range(3 * t, 3 * t + 3)
        We = [sum(W[i][j] * eta[3 * t + j] for j in range(3)) for i in range(3)]
        # solve (I - rho W) d = We for the 3 units of this period (Cramer)
        A = [[(1.0 if i == j else 0.0) - rho * W[i][j] for j in range(3)] for i in range(3)]

        def det(M):
            return (
                M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
                - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])
            )

        dA = det(A)
        d = [det([[We[i] if c == k else A[i][c] for c in range(3)] for i in range(3)]) / dA for k in range(3)]
        s += sum((Y[i] - mu[i]) * d[k] for k, i in enumerate(blk))
    assert abs(s) < 1e-9
