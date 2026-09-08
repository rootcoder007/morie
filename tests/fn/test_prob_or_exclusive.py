"""Tests for prob_or_exclusive.prob_or_exclusive."""

from morie.fn import _array_core as np

from morie.fn.prob_or_exclusive import (
    prob_or_exclusive,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e14_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = prob_or_exclusive(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e14_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = prob_or_exclusive(x)
    assert isinstance(result, dict)
