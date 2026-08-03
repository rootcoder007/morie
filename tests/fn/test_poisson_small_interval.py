"""Tests for poisson_small_interval.poisson_small_interval."""

from morie.fn import _array_core as np

from morie.fn.poisson_small_interval import (
    poisson_small_interval,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e18_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = poisson_small_interval(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e18_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = poisson_small_interval(x)
    assert isinstance(result, dict)
