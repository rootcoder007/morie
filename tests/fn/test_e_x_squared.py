"""Tests for e_x_squared.e_x_squared."""

from morie.fn import _array_core as np

from morie.fn.e_x_squared import (
    e_x_squared,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e70_basic():
    """Test basic functionality."""
    sigma = 2.0
    mu = 1.0
    result = e_x_squared(sigma, mu)
    assert "e_x2" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e70_edge():
    """Test edge cases."""
    result = e_x_squared(0.0, 0.0)
    assert "e_x2" in result
