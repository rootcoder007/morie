"""Tests for lnfrm.lognormal_frailty."""

import math

from morie.fn import _array_core as np

from morie.fn.lnfrm import lognormal_frailty


def test_lnfrm_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    p = 2
    time = np.linspace(0.1, 10, n)
    event = rng.integers(0, 2, n)
    X = rng.normal(0, 1, (n, p))
    cluster = rng.integers(0, 2, n)
    result = lognormal_frailty(time, event, X, cluster)
    assert "estimate" in result
    assert result["estimate"].shape == (p,)
    assert "se" in result
    assert result["se"].shape == (p,)
    assert "frailty" in result
    assert "sigma2" in result
    assert math.isfinite(result["sigma2"])
    assert result["sigma2"] >= 0.0
    assert "loglik_penalized" in result
    assert math.isfinite(result["loglik_penalized"])
    assert "n_outer" in result
    assert "n_newton" in result


def test_lnfrm_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 40
    p = 3
    time = np.linspace(0.1, 10, n)
    event = rng.integers(0, 2, n)
    X = rng.normal(0, 1, (n, p))
    cluster = rng.integers(0, 2, n)
    result = lognormal_frailty(time, event, X, cluster)
    assert "estimate" in result
    assert result["estimate"].shape == (p,)
    assert "se" in result
    assert result["se"].shape == (p,)
    assert "sigma2" in result
    assert math.isfinite(result["sigma2"])
    assert result["sigma2"] >= 0.0
