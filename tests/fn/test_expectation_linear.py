"""Tests for expectation_linear.expectation_linear."""

from morie.fn import _array_core as np

from morie.fn.expectation_linear import (
    expectation_linear,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e13_basic():
    """Test basic functionality."""
    a, e_x, b, e_y, c = 2.0, 3.0, 0.5, 4.0, 1.0
    result = expectation_linear(a, e_x, b, e_y, c)
    assert isinstance(result, dict)
    assert "expectation" in result
    assert result["expectation"] == a * e_x + b * e_y + c


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e13_edge():
    """Test edge cases."""
    a, e_x, b, e_y, c = 0.0, 5.0, 0.0, 7.0, 2.0
    result = expectation_linear(a, e_x, b, e_y, c)
    assert isinstance(result, dict)
    assert "expectation" in result
    assert result["expectation"] == c
