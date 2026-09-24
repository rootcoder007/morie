"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner2e95.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_95."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner2e95 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_95,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e95_basic():
    """Test basic functionality."""
    p = 0.5
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_95(p)
    assert isinstance(result, dict)
    assert "p_intersection" in result
    value = result["p_intersection"]
    assert math.isfinite(value)
    assert 0.0 <= value <= 1.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e95_edge():
    """Test edge cases."""
    p = 0.3
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_95(p, k=3)
    assert isinstance(result, dict)
    assert "p_intersection" in result
    value = result["p_intersection"]
    assert math.isfinite(value)
    assert 0.0 <= value <= 1.0
