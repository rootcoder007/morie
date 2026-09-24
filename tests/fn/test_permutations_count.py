"""Tests for permutations_count.permutations_count."""

from morie.fn import _array_core as np

from morie.fn.permutations_count import (
    permutations_count,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e3_basic():
    """Test basic functionality."""
    n = 5
    result = permutations_count(n)
    assert isinstance(result, dict)
    assert "n" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e3_edge():
    """Test edge cases."""
    n = 5
    result = permutations_count(n)
    assert isinstance(result, dict)
