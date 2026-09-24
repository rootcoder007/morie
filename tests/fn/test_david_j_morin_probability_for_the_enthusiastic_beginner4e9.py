"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner4e9.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_9."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner4e9 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_9,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e9_basic():
    """Test basic functionality."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_9(40)
    p = result.p
    assert isinstance(p, (int, float))
    assert math.isfinite(p)
    assert 0.0 <= p <= 1.0
    assert math.isclose(p, 1.0 / 41.0)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e9_edge():
    """Test edge cases."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_9(10)
    p = result.p
    assert isinstance(p, (int, float))
    assert math.isfinite(p)
    assert 0.0 <= p <= 1.0
    assert math.isclose(p, 1.0 / 11.0)
