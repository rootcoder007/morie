"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner1e4.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_4."""

import math

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner1e4 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_4,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e4_basic():
    """Test basic functionality."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_4(5, 3)
    # Result is a RichResult reporting N^n (ordered sampling with replacement)
    count = result["count"]
    assert isinstance(count, (int, float))
    assert math.isfinite(count)
    assert count == 5 ** 3


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e4_edge():
    """Test edge cases."""
    # N=1: only one possible ordered arrangement of any length
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_4(1, 5)
    count = result["count"]
    assert isinstance(count, (int, float))
    assert math.isfinite(count)
    assert count == 1
