"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner4e37.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_37."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner4e37 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_37,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e37_basic():
    """Test basic functionality."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_37(0.1, 5)
    assert isinstance(result, dict)
    assert "approx" in result
    assert math.isfinite(float(result["approx"]))
    assert 0.0 <= float(result["approx"]) <= 1.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e37_edge():
    """Test edge cases."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_37(0.0, 10)
    assert isinstance(result, dict)
    assert "approx" in result
    assert math.isfinite(float(result["approx"]))
    assert 0.0 <= float(result["approx"]) <= 1.0
