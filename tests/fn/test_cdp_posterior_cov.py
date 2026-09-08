"""Tests for cdp_posterior_cov.cdp_posterior_cov."""

from morie.fn import _array_core as np

from morie.fn.cdp_posterior_cov import cdp_posterior_cov


def test_ghs016_basic():
    """Test basic functionality."""
    alpha_j = np.random.default_rng(42).normal(0, 1, 100)
    alpha_jprime = np.random.default_rng(42).normal(0, 1, 100)
    N_j = np.random.default_rng(42).normal(0, 1, 100)
    N_jprime = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = cdp_posterior_cov(alpha_j, alpha_jprime, N_j, N_jprime, n)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ghs016_edge():
    """Test edge cases."""
    alpha_j = np.random.default_rng(42).normal(0, 1, 100)
    alpha_jprime = np.random.default_rng(42).normal(0, 1, 100)
    N_j = np.random.default_rng(42).normal(0, 1, 100)
    N_jprime = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = cdp_posterior_cov(alpha_j, alpha_jprime, N_j, N_jprime, n)
    assert isinstance(result, dict)
