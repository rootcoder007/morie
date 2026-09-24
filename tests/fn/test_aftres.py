"""Tests for aftres.aft_residuals."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.aftwbl import aft_weibull
from morie.fn.aftres import aft_residuals


def test_aftres_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 150
    p = 2
    X = rng.normal(0, 1, (n, p))
    # Generate Weibull-distributed times: log(T) = mu + sigma * Gumbel(0, 1)
    U = rng.uniform(0, 1, n)
    sigma = 0.5
    T = []
    for i in range(n):
        mu_i = 1.0 + 0.7 * X[i][0] - 0.4 * X[i][1]
        u = U[i] if U[i] > 1e-10 else 1e-10
        gumbel = -math.log(-math.log(u))
        T.append(math.exp(mu_i + sigma * gumbel))
    # Generate exponential censoring times
    U2 = rng.uniform(0, 1, n)
    C = []
    for i in range(n):
        u = U2[i] if U2[i] > 1e-10 else 1e-10
        C.append(-math.log(u) * 5.0)
    t = [min(T[i], C[i]) for i in range(n)]
    e = [1.0 if T[i] <= C[i] else 0.0 for i in range(n)]

    fit = aft_weibull(t, e, X)
    result = aft_residuals(fit)

    assert isinstance(result, dict)
    assert "standardized" in result
    assert "cox_snell" in result
    assert "martingale" in result
    assert "deviance" in result
    assert "family" in result
    # Cox-Snell residuals are non-negative by construction
    cs = result["cox_snell"]
    assert min(cs) >= 0.0
    # Martingale residuals are bounded above by 1
    mg = result["martingale"]
    assert max(mg) <= 1.0 + 1e-9


def test_aftres_edge():
    """Test edge cases - missing required keys in fit mapping."""
    with pytest.raises(ValueError):
        aft_residuals({"beta": [1.0]})
