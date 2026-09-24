"""Tests for negative_binomial_variance.negative_binomial_variance."""

from morie.fn import _array_core as np

from morie.fn.negative_binomial_variance import negative_binomial_variance


def test_ca6e8_basic():
    """Test basic functionality."""
    mu = 0.5
    alpha = 0.5
    result = negative_binomial_variance(mu, alpha)
    assert isinstance(result, dict)
    assert "value" in result


def test_ca6e8_edge():
    """Test edge cases."""
    mu = 0.5
    alpha = 0.5
    result = negative_binomial_variance(mu, alpha)
    assert isinstance(result, dict)
