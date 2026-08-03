"""Tests for exponential_waiting_density.exponential_waiting_density."""

from morie.fn import _array_core as np

from morie.fn.exponential_waiting_density import (
    exponential_waiting_density,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e26_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = exponential_waiting_density(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e26_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = exponential_waiting_density(x)
    assert isinstance(result, dict)
