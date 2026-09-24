"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner3e12.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_12."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner3e12 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_12,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e12_basic():
    """Test basic functionality."""
    values_x = [0, 1, 2]
    probs_x = [1.0/3.0, 1.0/3.0, 1.0/3.0]
    values_y = [0, 1]
    probs_y = [0.5, 0.5]
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_12(values_x, probs_x, values_y, probs_y)
    assert isinstance(result, dict)
    assert "e_sum" in result
    assert "e_x_plus_e_y" in result
    assert math.isfinite(result["e_sum"])
    assert math.isfinite(result["e_x_plus_e_y"])


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e12_edge():
    """Test edge cases."""
    values_x = [1]
    probs_x = [1.0]
    values_y = [5]
    probs_y = [1.0]
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_12(values_x, probs_x, values_y, probs_y)
    assert isinstance(result, dict)
    assert "e_sum" in result
    assert "e_x_plus_e_y" in result
    assert math.isfinite(result["e_sum"])
    assert math.isfinite(result["e_x_plus_e_y"])
