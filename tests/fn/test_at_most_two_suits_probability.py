"""Tests for at_most_two_suits_probability.at_most_two_suits_probability."""

import math

from morie.fn import _array_core as np

from morie.fn.at_most_two_suits_probability import (
    at_most_two_suits_probability,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e43_basic():
    """Test basic functionality."""
    result = at_most_two_suits_probability(n_suits=4, n_ranks=13, hand=5)
    assert isinstance(result, dict)
    assert "favorable" in result
    assert "total" in result
    assert "probability" in result
    assert math.isfinite(result["probability"])
    assert 0.0 <= result["probability"] <= 1.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e43_edge():
    """Test edge cases."""
    result = at_most_two_suits_probability(n_suits=2, n_ranks=2, hand=2)
    assert isinstance(result, dict)
    assert "favorable" in result
    assert "total" in result
    assert "probability" in result
    assert math.isfinite(result["probability"])
    assert 0.0 <= result["probability"] <= 1.0
