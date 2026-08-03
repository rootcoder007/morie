"""Tests for excess_score_factor.excess_score_factor."""

from morie.fn import _array_core as np

from morie.fn.excess_score_factor import (
    excess_score_factor,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e81_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = excess_score_factor(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e81_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = excess_score_factor(x)
    assert isinstance(result, dict)
