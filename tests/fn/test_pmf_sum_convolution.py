"""Tests for pmf_sum_convolution.pmf_sum_convolution."""

from morie.fn import _array_core as np

from morie.fn.pmf_sum_convolution import (
    pmf_sum_convolution,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e11_basic():
    """Test basic functionality."""
    values_x = 0.5
    probs_x = 1
    values_y = 0.5
    probs_y = 1
    result = pmf_sum_convolution(values_x, probs_x, values_y, probs_y)
    assert isinstance(result, dict)
    assert "values" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e11_edge():
    """Test edge cases."""
    values_x = 0.5
    probs_x = 1
    values_y = 0.5
    probs_y = 1
    result = pmf_sum_convolution(values_x, probs_x, values_y, probs_y)
    assert isinstance(result, dict)
