"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner5e16.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_16."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner5e16 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_16,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner5e16_basic():
    """Test basic functionality."""
    k = 10
    a = 5.0
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_16(k, a)
    assert isinstance(result, dict)
    assert "approx" in result
    assert "exact" in result
    assert "rel_error" in result
    assert math.isfinite(float(result["approx"]))
    assert math.isfinite(float(result["exact"]))
    assert math.isfinite(float(result["rel_error"]))


def test_david_j_morin_probability_for_the_enthusiastic_beginner5e16_edge():
    """Test edge cases."""
    k = 1
    a = 0.5
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_16(k, a)
    assert isinstance(result, dict)
    assert "approx" in result
    assert "exact" in result
    assert "rel_error" in result
    assert math.isfinite(float(result["approx"]))
    assert math.isfinite(float(result["exact"]))
    assert math.isfinite(float(result["rel_error"]))
