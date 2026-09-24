"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner4e66.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_66."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner4e66 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_66,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e66_basic():
    """Test basic functionality."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_66(10, 0.5)
    assert isinstance(result, dict)
    assert "second_moment" in result
    assert math.isfinite(result["second_moment"])


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e66_edge():
    """Test edge cases."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_66(1, 0.0)
    assert isinstance(result, dict)
    assert "second_moment" in result
    assert math.isfinite(result["second_moment"])
