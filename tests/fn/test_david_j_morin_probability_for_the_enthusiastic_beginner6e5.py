"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner6e5.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_5."""

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner6e5 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_5,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e5_basic():
    """Test basic functionality."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_5(2.0, 1.0, 0.5)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e5_edge():
    """Test edge cases."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_5(0.5, 0.1, 0.1)
    assert isinstance(result, dict)
    assert len(result) > 0
