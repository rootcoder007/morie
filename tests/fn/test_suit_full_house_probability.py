"""Tests for suit_full_house_probability.suit_full_house_probability."""

from morie.fn import _array_core as np

from morie.fn.suit_full_house_probability import (
    suit_full_house_probability,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e41_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = suit_full_house_probability(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e41_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = suit_full_house_probability(x)
    assert isinstance(result, dict)
