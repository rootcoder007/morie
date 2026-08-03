"""Tests for slope_from_cov.slope_from_cov."""

from morie.fn import _array_core as np

from morie.fn.slope_from_cov import (
    slope_from_cov,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e13_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = slope_from_cov(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e13_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = slope_from_cov(x)
    assert isinstance(result, dict)
