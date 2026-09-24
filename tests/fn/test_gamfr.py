"""Tests for gamfr.gamma_frailty_cox."""

import math

from morie.fn import _array_core as np

from morie.fn.gamfr import gamma_frailty_cox


def test_gamfr_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    p = 3
    n_clusters = 8
    X = rng.normal(0, 1, (n, p))
    time = rng.uniform(0.1, 10.0, n)
    event = [float(x) for x in rng.integers(0, 2, n)]
    cluster = rng.integers(0, n_clusters, n)
    result = gamma_frailty_cox(time, event, X, cluster)
    assert isinstance(result, dict)
    beta = result["beta"]
    assert len(beta) == p
    se = result["se"]
    assert len(se) == p
    theta = result["theta"]
    assert math.isfinite(theta)
    assert theta > 0
    kt = result["kendall_tau"]
    assert 0.0 <= kt < 1.0
    assert result["marginal_attenuation"] is True


def test_gamfr_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 24
    p = 2
    n_clusters = 4
    X = rng.normal(0, 1, (n, p))
    time = rng.uniform(0.1, 10.0, n)
    event = [float(x) for x in rng.integers(0, 2, n)]
    cluster = rng.integers(0, n_clusters, n)
    result = gamma_frailty_cox(time, event, X, cluster)
    assert isinstance(result, dict)
    assert "beta" in result
    assert "theta" in result
    assert "kendall_tau" in result
    assert "frailty" in result
    assert "marginal_attenuation" in result
