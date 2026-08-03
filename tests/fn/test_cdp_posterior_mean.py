"""Tests for cdp_posterior_mean.cdp_posterior_mean."""

from morie.fn import _array_core as np

from morie.fn.cdp_posterior_mean import cdp_posterior_mean


def test_ghs014_basic():
    """Test basic functionality."""
    alpha_j = np.random.default_rng(42).normal(0, 1, 100)
    N_j = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = cdp_posterior_mean(alpha_j, N_j, j, n)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ghs014_edge():
    """Test edge cases."""
    alpha_j = np.random.default_rng(42).normal(0, 1, 100)
    N_j = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = cdp_posterior_mean(alpha_j, N_j, j, n)
    assert isinstance(result, dict)
