"""Tests for var_sum_with_cov.var_sum_with_cov."""

from morie.fn import _array_core as np

from morie.fn.var_sum_with_cov import (
    var_sum_with_cov,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e26_basic():
    """Test basic functionality."""
    var_x = 0.5
    var_y = 0.5
    result = var_sum_with_cov(var_x, var_y)
    assert isinstance(result, dict)
    assert "var_sum" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e26_edge():
    """Test edge cases."""
    var_x = 0.5
    var_y = 0.5
    result = var_sum_with_cov(var_x, var_y)
    assert isinstance(result, dict)
