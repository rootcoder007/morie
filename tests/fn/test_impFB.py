"""Tests for impFB (Hu, Koren & Volinsky 2008, implicit-feedback ALS)."""

import pytest

from morie.fn.impFB import (als_step, confidence, cost, explain,
                            implicit_feedback_loss, preference)

R = [[0, 3, 0, 1, 0], [2, 0, 0, 0, 5], [0, 1, 4, 0, 0], [1, 0, 0, 2, 0]]


def test_impFB_basic():
    """Eq. (4): the fast Y'Y + Y'(C-I)Y route equals Y'CY, and the step
    solves (Y'C^uY + lam I) x = Y'C^u p(u); ALS never raises eq. (3)."""
    Y = [[0.3, -0.1], [0.2, 0.4], [-0.5, 0.1], [0.1, 0.1], [0.6, -0.2]]
    C = confidence(R, 40.0)[1]
    P = preference(R)[1]
    fast = als_step(Y, C, P, 0.1, fast=True)
    slow = als_step(Y, C, P, 0.1, fast=False)
    assert fast == pytest.approx(slow, rel=1e-12)
    for a in range(2):
        lhs = sum(sum(C[i] * Y[i][a] * Y[i][b] for i in range(5)) * fast[b]
                  for b in range(2)) + 0.1 * fast[a]
        assert lhs == pytest.approx(sum(C[i] * P[i] * Y[i][a] for i in range(5)), rel=1e-12)
    r = implicit_feedback_loss(R, f=2, iters=8, seed=1)
    h = r["cost_history"]
    assert all(b <= a * (1 + 1e-12) for a, b in zip(h, h[1:]))
    assert r["final_cost"] == pytest.approx(cost(R, r["X"], r["Y"], 40.0, 0.1), rel=1e-12)


def test_impFB_edge():
    """p = 1[r > 0], c = 1 + alpha r; Sec. 5's contributions sum to x_u'y_i."""
    assert preference([[0, 2.5, -1]]) == [[0.0, 1.0, 0.0]]
    assert confidence([[0, 2]], alpha=10) == [[1.0, 21.0]]
    r = implicit_feedback_loss(R, f=2, iters=5, seed=0)
    Y = r["Y"]
    C, P = confidence(R)[0], preference(R)[0]
    x = als_step(Y, C, P, 0.1)
    e = explain(Y, C, P, 2, lam=0.1)
    assert e["prediction"] == pytest.approx(sum(a * b for a, b in zip(x, Y[2])), rel=1e-10)
    assert set(e["contributions"]) == {1, 3}
    with pytest.raises(ValueError, match="negative"):
        implicit_feedback_loss([[1, -2]])
    with pytest.raises(ValueError, match="at least 1"):
        implicit_feedback_loss(R, f=0)
