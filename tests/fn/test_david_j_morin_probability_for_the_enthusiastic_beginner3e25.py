"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner3e25.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_25."""

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner3e25 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_25,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e25_basic():
    """Test basic functionality."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_25(1.0, 2.0)
    assert isinstance(result, dict)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e25_edge():
    """Test edge cases."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_25(0.5, 0.0)
    assert isinstance(result, dict)
