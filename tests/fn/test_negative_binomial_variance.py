"""Tests for negative_binomial_variance.negative_binomial_variance."""

from morie.fn import _array_core as np

from morie.fn.negative_binomial_variance import negative_binomial_variance


def test_ca6e8_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = negative_binomial_variance(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca6e8_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = negative_binomial_variance(x)
    assert isinstance(result, dict)
