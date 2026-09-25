"""Tests for lmmf1.lmm_form_eq2_1."""

import pytest

from morie.fn.lmmf1 import lmm_form_eq2_1


def test_lmmf1_basic():
    """Y = X beta + Z u + e: E[Y] = X beta, E[Y | u] = X beta + Z u and
    Var(Y) = Z Sigma Z' + R (R = I by default)."""
    X = [[1.0, 0.5], [1.0, -1.0], [1.0, 2.0]]
    Zm = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    beta, u = [2.0, 0.3], [0.4, -0.7]
    S = [[1.5, 0.2], [0.2, 0.8]]
    r = lmm_form_eq2_1(X, beta, Zm, u, S)
    xb = [sum(a * b for a, b in zip(row, beta)) for row in X]
    zu = [sum(a * b for a, b in zip(row, u)) for row in Zm]
    assert list(r["mean_marginal"]) == pytest.approx(xb, rel=1e-15)
    assert list(r["mean_conditional"]) == pytest.approx([a + b for a, b in zip(xb, zu)], rel=1e-15)
    V = [list(v) for v in (r["V"].tolist() if hasattr(r["V"], "tolist") else r["V"])]
    for i in range(3):
        for j in range(3):
            want = sum(Zm[i][a] * S[a][b] * Zm[j][b] for a in range(2) for b in range(2)) + (i == j)
            assert V[i][j] == pytest.approx(want, rel=1e-15)


def test_lmmf1_edge():
    """A supplied residual covariance replaces the identity."""
    r = lmm_form_eq2_1([[1.0]], [1.0], [[1.0]], [0.0], [[2.0]], R=[[0.5]])
    V = r["V"].tolist() if hasattr(r["V"], "tolist") else r["V"]
    assert V[0][0] == pytest.approx(2.5, rel=1e-15)


