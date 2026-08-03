"""Tests for gaussian_sum_density.gaussian_sum_density."""

from morie.fn import _array_core as np

from morie.fn.gaussian_sum_density import (
    gaussian_sum_density,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e70_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = gaussian_sum_density(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e70_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = gaussian_sum_density(x)
    assert isinstance(result, dict)
