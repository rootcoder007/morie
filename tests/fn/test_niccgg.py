"""Tests for niccgg.nakagawa_marginal_r2."""

import math

from morie.fn import _array_core as np

from morie.fn.niccgg import nakagawa_marginal_r2


def test_niccgg_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    rng_X = np.random.default_rng(42)
    rng_c = np.random.default_rng(44)

    N = 100
    k = 5

    y = rng_y.normal(0, 1, N)
    X = rng_X.normal(0, 1, (N, 3))
    cluster = rng_c.integers(0, k, N)

    result = nakagawa_marginal_r2(y, X, cluster=cluster)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "r2_marginal" in result
    assert "r2_conditional" in result
    assert "sigma2_f" in result
    assert "sigma2_l" in result
    assert "sigma2_e" in result
    assert "icc" in result
    assert "n" in result
    assert "n_clusters" in result

    assert math.isfinite(result["r2_marginal"])
    assert 0 <= result["r2_marginal"] <= 1
    assert math.isfinite(result["r2_conditional"])
    assert 0 <= result["r2_conditional"] <= 1
    assert result["n"] == N
    assert result["n_clusters"] == k


def test_niccgg_edge():
    """Test edge cases."""
    rng_y = np.random.default_rng(43)
    rng_X = np.random.default_rng(42)
    rng_c = np.random.default_rng(44)

    N = 40
    k = 3

    y = rng_y.normal(0, 1, N)
    X = rng_X.normal(0, 1, (N, 3))
    cluster = rng_c.integers(0, k, N)

    result = nakagawa_marginal_r2(y, X, cluster=cluster)
    assert isinstance(result, dict)
    assert "r2_marginal" in result
    assert "r2_conditional" in result
    assert result["n"] == N
    assert result["n_clusters"] == k
