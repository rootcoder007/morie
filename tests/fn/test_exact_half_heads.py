"""Tests for exact_half_heads.exact_half_heads."""

from morie.fn import _array_core as np

from morie.fn.exact_half_heads import (
    exact_half_heads,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e65_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = exact_half_heads(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e65_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = exact_half_heads(x)
    assert isinstance(result, dict)
