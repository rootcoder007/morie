"""Tests for density_expectation.density_expectation."""

from morie.fn import _array_core as np

from morie.fn.density_expectation import (
    density_expectation,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e55_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = density_expectation(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e55_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = density_expectation(x)
    assert isinstance(result, dict)
