"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner3e15.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_15."""

import math

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner3e15 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_15,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e15_basic():
    """Test basic functionality."""
    e_x = 3.5
    n = 10
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_15(e_x, n)
    assert isinstance(result, dict)
    assert math.isfinite(result["e_sum"])
    assert result["n"] == n
    assert result["e_sum"] == e_x * n


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e15_edge():
    """Test edge cases."""
    e_x = -2.0
    n = 5
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_15(e_x, n)
    assert isinstance(result, dict)
    assert math.isfinite(result["e_sum"])
    assert result["n"] == n
    assert result["e_sum"] == e_x * n
