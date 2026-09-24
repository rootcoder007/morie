"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner7e5.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_5."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner7e5 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_5,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner7e5_basic():
    """Test basic functionality."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_5(1.0, 2)
    assert isinstance(result, dict)
    assert "ratio" in result
    assert "well_inside" in result
    assert math.isfinite(float(result["ratio"]))
    assert float(result["ratio"]) >= 0.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner7e5_edge():
    """Test edge cases."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_5(0.0, 1)
    assert isinstance(result, dict)
    assert "ratio" in result
    assert "well_inside" in result
    assert math.isfinite(float(result["ratio"]))
    assert float(result["ratio"]) >= 0.0
