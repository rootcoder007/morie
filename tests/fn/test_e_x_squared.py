"""Tests for e_x_squared.e_x_squared."""

from morie.fn import _array_core as np

from morie.fn.e_x_squared import (
    e_x_squared,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e70_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = e_x_squared(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e70_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = e_x_squared(x)
    assert isinstance(result, dict)
