"""Tests for prob_or_general.prob_or_general."""

from morie.fn import _array_core as np

from morie.fn.prob_or_general import (
    prob_or_general,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e21_basic():
    """Test basic functionality."""
    p_a = 0.5
    p_b = 0.5
    p_ab = 0.5
    result = prob_or_general(p_a, p_b, p_ab)
    assert isinstance(result, dict)
    assert "p_a" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e21_edge():
    """Test edge cases."""
    p_a = 0.5
    p_b = 0.5
    p_ab = 0.5
    result = prob_or_general(p_a, p_b, p_ab)
    assert isinstance(result, dict)
