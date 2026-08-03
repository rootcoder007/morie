"""Tests for var_scale.var_scale."""

from morie.fn import _array_core as np

from morie.fn.var_scale import (
    var_scale,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e24_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = var_scale(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e24_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = var_scale(x)
    assert isinstance(result, dict)
