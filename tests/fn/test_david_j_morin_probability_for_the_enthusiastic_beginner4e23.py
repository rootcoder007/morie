"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner4e23.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_23."""

import math

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner4e23 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_23,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e23_basic():
    """Test basic functionality."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_23(1.0, 0.5, 0.5)
    assert isinstance(result, dict)
    assert "probability" in result
    p = result["probability"]
    assert math.isfinite(p)
    assert 0.0 <= p <= 1.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e23_edge():
    """Test edge cases."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_23(0.0, 0.5, 1.0)
    assert isinstance(result, dict)
    assert "probability" in result
    p = result["probability"]
    assert math.isfinite(p)
    assert 0.0 <= p <= 1.0
