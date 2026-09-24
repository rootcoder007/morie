"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner7e11.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_11."""

import math

from morie.fn import _array_core as np
from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner7e11 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_11,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner7e11_basic():
    """Test basic functionality."""
    a = 5.0
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_11(a)
    assert isinstance(result, dict)
    assert "total" in result
    value = result["total"]
    assert math.isfinite(value)
    assert 0 <= value <= 1


def test_david_j_morin_probability_for_the_enthusiastic_beginner7e11_edge():
    """Test edge cases."""
    a = 0.5
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_11(a)
    assert isinstance(result, dict)
    assert "total" in result
    value = result["total"]
    assert math.isfinite(value)
    assert 0 <= value <= 1
