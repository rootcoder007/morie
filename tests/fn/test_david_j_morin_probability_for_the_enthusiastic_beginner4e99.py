"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner4e99.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_99."""

import math

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner4e99 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_99,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e99_basic():
    """Test basic functionality."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_99(2.5)
    assert 'p_zero' in result.payload
    assert 'a' in result.payload
    assert result.payload['a'] == 2.5
    p = result.p_zero
    assert math.isfinite(p)
    assert 0.0 <= p <= 1.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e99_edge():
    """Test edge cases."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_99(0.0)
    assert 'p_zero' in result.payload
    assert 'a' in result.payload
    assert result.payload['a'] == 0.0
    p = result.p_zero
    assert math.isfinite(p)
    assert 0.0 <= p <= 1.0
