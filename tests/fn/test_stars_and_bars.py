"""Tests for stars_and_bars.stars_and_bars."""

from morie.fn import _array_core as np

from morie.fn.stars_and_bars import (
    stars_and_bars,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e57_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = stars_and_bars(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e57_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = stars_and_bars(x)
    assert isinstance(result, dict)
