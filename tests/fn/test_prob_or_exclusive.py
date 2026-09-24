"""Tests for prob_or_exclusive.prob_or_exclusive."""

from morie.fn import _array_core as np

from morie.fn.prob_or_exclusive import (
    prob_or_exclusive,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e14_basic():
    """Test basic functionality."""
    ps = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = prob_or_exclusive(ps)
    assert isinstance(result, dict)
    assert "ps" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e14_edge():
    """Test edge cases."""
    ps = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = prob_or_exclusive(ps)
    assert isinstance(result, dict)
