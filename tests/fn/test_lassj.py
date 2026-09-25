"""Tests for lassj.lasso_objective."""

import pytest

from morie.fn.lassj import lasso_objective

X = [[1.0, -0.5], [0.3, 2.0], [-1.2, 0.7], [0.8, 0.1]]
Y = [2.0, 1.1, -0.9, 1.4]


def test_lassj_basic():
    """PRSS = sum (y - b0 - x b)^2 + lam sum |b_j| with the intercept
    unpenalised, and S(b, lam) = sign(b) max(|b| - lam, 0), recomputed."""
    beta, lam = [0.4, 1.3, -0.2], 0.5
    result = lasso_objective(X, Y, beta, lam)
    assert isinstance(result, dict)
    rss = sum((y - beta[0] - x[0] * beta[1] - x[1] * beta[2]) ** 2 for x, y in zip(X, Y))
    assert result["rss"] == pytest.approx(rss, rel=1e-14)
    assert result["penalty"] == pytest.approx(lam * (1.3 + 0.2), rel=1e-15)
    assert result["prss"] == pytest.approx(rss + lam * 1.5, rel=1e-14)
    soft = [(1 if b > 0 else -1) * max(abs(b) - lam, 0.0) for b in beta[1:]]
    assert list(result["soft"])[-2:] == pytest.approx(soft, abs=1e-15)


def test_lassj_edge():
    """Without an intercept every coefficient is penalised; a negative
    lambda and a row-count mismatch are refused."""
    r = lasso_objective(X, Y, [1.3, -0.2], 2.0, add_intercept=False)
    assert r["penalty"] == pytest.approx(2.0 * 1.5, rel=1e-15)
    with pytest.raises(ValueError):
        lasso_objective(X, Y, [1.3, -0.2], -1.0, add_intercept=False)
    with pytest.raises(ValueError):
        lasso_objective(X[:3], Y, [0.0, 1.3, -0.2], 0.1)
