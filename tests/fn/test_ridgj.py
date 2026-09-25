"""Tests for ridgj.ridge_objective (MVSML sec. 3.6.1)."""

import pytest

from morie.fn.ridgj import ridge_objective


X = [[1.0, 2.0], [0.5, -1.0], [2.0, 0.0], [-1.0, 1.5]]
Y = [3.0, 0.2, 2.5, 0.1]


def test_ridgj_basic():
    """PRSS = sum (y - b0 - x'b)^2 + lambda sum_{j >= 1} b_j^2: the
    intercept is not penalised; without an intercept every coefficient is."""
    b = [0.4, 1.1, 0.6]
    r = ridge_objective(X, Y, b, 2.0)
    rss = sum((y - b[0] - x[0] * b[1] - x[1] * b[2]) ** 2 for x, y in zip(X, Y))
    assert r["rss"] == pytest.approx(rss, rel=1e-15)
    assert r["penalty"] == pytest.approx(2.0 * (1.1 ** 2 + 0.6 ** 2), rel=1e-15)
    assert r["prss"] == pytest.approx(rss + r["penalty"], rel=1e-15)
    assert (r["n"], r["p"]) == (4, 3)
    n = ridge_objective(X, Y, [1.1, 0.6], 2.0, add_intercept=False)
    assert n["penalty"] == pytest.approx(2.0 * (1.1 ** 2 + 0.6 ** 2), rel=1e-15)


def test_ridgj_edge():
    """Negative lambda, a short y and a wrong-length beta raise."""
    with pytest.raises(ValueError):
        ridge_objective(X, Y, [0.0, 0.0, 0.0], -1.0)
    with pytest.raises(ValueError):
        ridge_objective(X, Y[:3], [0.0, 0.0, 0.0], 1.0)
    with pytest.raises(ValueError):
        ridge_objective(X, Y, [0.0, 0.0], 1.0)
