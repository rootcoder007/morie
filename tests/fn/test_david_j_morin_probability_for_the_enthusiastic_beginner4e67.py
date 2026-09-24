"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner4e67.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_67."""

import math

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner4e67 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_67,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e67_basic():
    """Test basic functionality."""
    n = 40
    p = 0.3
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_67(n, p)
    assert isinstance(result, dict)
    assert "variance" in result
    variance = result["variance"]
    assert math.isfinite(variance)
    assert 0 <= variance <= n / 4


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e67_edge():
    """Test edge cases."""
    n = 10
    p = 0.5
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_67(n, p)
    assert isinstance(result, dict)
    assert "variance" in result
    variance = result["variance"]
    assert math.isfinite(variance)
    assert variance == n * p * (1 - p)
