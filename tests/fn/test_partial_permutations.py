"""Tests for partial_permutations.partial_permutations."""

from morie.fn import _array_core as np

from morie.fn.partial_permutations import (
    partial_permutations,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e5_basic():
    """Test basic functionality."""
    N = 5
    n = 5
    result = partial_permutations(N, n)
    assert isinstance(result, dict)
    assert "N" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e5_edge():
    """Test edge cases."""
    N = 5
    n = 5
    result = partial_permutations(N, n)
    assert isinstance(result, dict)
