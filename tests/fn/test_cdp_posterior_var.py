"""Tests for cdp_posterior_var.cdp_posterior_var."""

from morie.fn import _array_core as np

from morie.fn.cdp_posterior_var import cdp_posterior_var


def test_ghs015_basic():
    """Test basic functionality."""
    alpha_j = np.random.default_rng(42).normal(0, 1, 100)
    N_j = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = cdp_posterior_var(alpha_j, N_j, j, n)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ghs015_edge():
    """Test edge cases."""
    alpha_j = np.random.default_rng(42).normal(0, 1, 100)
    N_j = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = cdp_posterior_var(alpha_j, N_j, j, n)
    assert isinstance(result, dict)
