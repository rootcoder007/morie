"""Tests for exponential_crossing_time.exponential_crossing_time."""

from morie.fn import _array_core as np

from morie.fn.exponential_crossing_time import (
    exponential_crossing_time,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e30_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = exponential_crossing_time(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e30_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = exponential_crossing_time(x)
    assert isinstance(result, dict)
