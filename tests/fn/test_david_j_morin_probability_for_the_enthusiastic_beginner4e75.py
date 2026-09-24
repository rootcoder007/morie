"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner4e75.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_75."""

import math

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner4e75 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_75,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e75_basic():
    """Test basic functionality."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_75(
        k=5, n=20, p=0.3, N=100
    )
    assert isinstance(result, dict)
    assert len(result) > 0
    has_finite = any(
        isinstance(v, (int, float)) and math.isfinite(v)
        for v in result.values()
    )
    assert has_finite


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e75_edge():
    """Test edge cases."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_75(
        k=0, n=3, p=0.5, N=10
    )
    assert isinstance(result, dict)
    assert len(result) > 0
    has_finite = any(
        isinstance(v, (int, float)) and math.isfinite(v)
        for v in result.values()
    )
    assert has_finite
