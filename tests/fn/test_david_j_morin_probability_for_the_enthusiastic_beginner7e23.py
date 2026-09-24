"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner7e23.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_23."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner7e23 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_23,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner7e23_basic():
    """Test basic functionality."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_23(2.0, 5)
    assert isinstance(result, dict)
    for key in ("exact", "approx", "na2", "valid"):
        assert key in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner7e23_edge():
    """Test edge cases."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_23(0.5, 1)
    assert isinstance(result, dict)
    for key in ("exact", "approx", "na2", "valid"):
        assert key in result
