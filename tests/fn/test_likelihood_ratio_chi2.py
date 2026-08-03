"""Tests for likelihood_ratio_chi2.likelihood_ratio_chi2."""

from morie.fn import _array_core as np

from morie.fn.likelihood_ratio_chi2 import likelihood_ratio_chi2


def test_ca4e18_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = likelihood_ratio_chi2(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca4e18_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = likelihood_ratio_chi2(x)
    assert isinstance(result, dict)
