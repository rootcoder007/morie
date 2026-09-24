"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner3e20.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_20."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner3e20 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_20,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e20_basic():
    """Test basic functionality."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_20(sides=6)
    assert hasattr(result, "mean")
    assert hasattr(result, "variance")
    assert math.isfinite(float(result.mean))
    assert math.isfinite(float(result.variance))
    assert float(result.variance) >= 0


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e20_edge():
    """Test edge cases."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_20(sides=2)
    assert hasattr(result, "mean")
    assert hasattr(result, "variance")
    assert math.isfinite(float(result.mean))
    assert math.isfinite(float(result.variance))
    assert float(result.variance) >= 0
