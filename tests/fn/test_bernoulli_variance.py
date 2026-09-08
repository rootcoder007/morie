"""Tests for bernoulli_variance.bernoulli_variance."""

from morie.fn import _array_core as np

from morie.fn.bernoulli_variance import (
    bernoulli_variance,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e22_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bernoulli_variance(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e22_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bernoulli_variance(x)
    assert isinstance(result, dict)
