"""Tests for beta_standardized.beta_standardized."""

from morie.fn import _array_core as np

from morie.fn.beta_standardized import beta_standardized


def test_ca2e20_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = beta_standardized(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca2e20_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = beta_standardized(x)
    assert isinstance(result, dict)
