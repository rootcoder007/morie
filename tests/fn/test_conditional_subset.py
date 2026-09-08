"""Tests for conditional_subset.conditional_subset."""

from morie.fn import _array_core as np

from morie.fn.conditional_subset import (
    conditional_subset,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e49_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = conditional_subset(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e49_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = conditional_subset(x)
    assert isinstance(result, dict)
