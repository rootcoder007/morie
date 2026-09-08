"""Tests for bayes_general.bayes_general."""

from morie.fn import _array_core as np

from morie.fn.bayes_general import (
    bayes_general,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e74_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_general(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e74_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_general(x)
    assert isinstance(result, dict)
