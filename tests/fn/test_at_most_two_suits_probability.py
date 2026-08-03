"""Tests for at_most_two_suits_probability.at_most_two_suits_probability."""

from morie.fn import _array_core as np

from morie.fn.at_most_two_suits_probability import (
    at_most_two_suits_probability,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e43_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = at_most_two_suits_probability(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e43_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = at_most_two_suits_probability(x)
    assert isinstance(result, dict)
