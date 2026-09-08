"""Tests for binomial_variance.binomial_variance."""

from morie.fn import _array_core as np

from morie.fn.binomial_variance import (
    binomial_variance,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e33_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = binomial_variance(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e33_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = binomial_variance(x)
    assert isinstance(result, dict)
