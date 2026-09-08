"""Tests for factorial.factorial."""

from morie.fn import _array_core as np

from morie.fn.factorial import (
    factorial,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = factorial(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = factorial(x)
    assert isinstance(result, dict)
