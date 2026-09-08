"""Tests for regression_to_mean_factor.regression_to_mean_factor."""

from morie.fn import _array_core as np

from morie.fn.regression_to_mean_factor import (
    regression_to_mean_factor,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e40_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = regression_to_mean_factor(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e40_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = regression_to_mean_factor(x)
    assert isinstance(result, dict)
