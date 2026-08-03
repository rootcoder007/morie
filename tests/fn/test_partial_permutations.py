"""Tests for partial_permutations.partial_permutations."""

from morie.fn import _array_core as np

from morie.fn.partial_permutations import (
    partial_permutations,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e5_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = partial_permutations(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e5_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = partial_permutations(x)
    assert isinstance(result, dict)
