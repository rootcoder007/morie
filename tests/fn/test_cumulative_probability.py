"""Tests for cumulative_probability.cumulative_probability."""

from morie.fn import _array_core as np

from morie.fn.cumulative_probability import cumulative_probability


def test_ca5e6_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cumulative_probability(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca5e6_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cumulative_probability(x)
    assert isinstance(result, dict)
