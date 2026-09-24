"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner4e61.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_61."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner4e61 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_61,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e61_basic():
    """Test basic functionality."""
    n, p = 20, 0.5
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_61(n, p)
    assert isinstance(result, dict)
    assert "mean" in result
    assert math.isfinite(result["mean"])
    assert result["mean"] == n * p


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e61_edge():
    """Test edge cases."""
    n, p = 1, 0.0
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_61(n, p)
    assert isinstance(result, dict)
    assert "mean" in result
    assert math.isfinite(result["mean"])
    assert result["mean"] == n * p
