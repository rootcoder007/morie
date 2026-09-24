"""Tests for hockey_stick.hockey_stick."""

from morie.fn import _array_core as np

from morie.fn.hockey_stick import (
    hockey_stick,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e29_basic():
    """Test basic functionality."""
    n = 5
    k = 5
    result = hockey_stick(n, k)
    assert isinstance(result, dict)
    assert "n" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e29_edge():
    """Test edge cases."""
    n = 5
    k = 5
    result = hockey_stick(n, k)
    assert isinstance(result, dict)
