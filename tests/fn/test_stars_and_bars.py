"""Tests for stars_and_bars.stars_and_bars."""

from morie.fn import _array_core as np

from morie.fn.stars_and_bars import (
    stars_and_bars,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e57_basic():
    """Test basic functionality."""
    n = 5
    N = 5
    result = stars_and_bars(n, N)
    assert isinstance(result, dict)
    assert "n" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e57_edge():
    """Test edge cases."""
    n = 5
    N = 5
    result = stars_and_bars(n, N)
    assert isinstance(result, dict)
