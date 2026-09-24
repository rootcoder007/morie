"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner6e17.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_17."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner6e17 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_17,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e17_basic():
    """Test basic functionality."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_17(1.0, 1.0, 1.0)
    assert isinstance(result, dict)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e17_edge():
    """Test edge cases."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_17(0.5, 0.1, 2.0)
    assert isinstance(result, dict)
