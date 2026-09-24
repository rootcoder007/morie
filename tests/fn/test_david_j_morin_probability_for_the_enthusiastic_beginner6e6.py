"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner6e6.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_6."""
import math

from morie.fn import _array_core as np
from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner6e6 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_6,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e6_basic():
    """Test basic functionality."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_6(2.0, 1.0, 1.0)
    assert isinstance(result, dict)
    assert len(result) > 0
    numeric_values = [v for v in result.values() if isinstance(v, (int, float))]
    assert numeric_values
    assert all(math.isfinite(v) for v in numeric_values)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e6_edge():
    """Test edge cases with small but valid parameter values."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_6(0.5, 0.5, 0.5)
    assert isinstance(result, dict)
    assert len(result) > 0
    numeric_values = [v for v in result.values() if isinstance(v, (int, float))]
    assert all(math.isfinite(v) for v in numeric_values)
