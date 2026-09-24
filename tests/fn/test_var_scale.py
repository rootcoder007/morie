"""Tests for var_scale.var_scale."""

from morie.fn import _array_core as np

from morie.fn.var_scale import (
    var_scale,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e24_basic():
    """Test basic functionality."""
    a = 0.5
    var_x = 0.5
    result = var_scale(a, var_x)
    assert isinstance(result, dict)
    assert "a" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e24_edge():
    """Test edge cases."""
    a = 0.5
    var_x = 0.5
    result = var_scale(a, var_x)
    assert isinstance(result, dict)
