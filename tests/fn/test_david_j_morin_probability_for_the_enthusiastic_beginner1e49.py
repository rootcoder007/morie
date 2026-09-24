"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner1e49.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_49."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner1e49 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_49,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e49_basic():
    """Test basic functionality."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_49(3, 5)
    assert isinstance(result, dict)
    assert len(result) > 0
    first_val = next(iter(result.values()))
    assert isinstance(first_val, (int, float))
    assert math.isfinite(first_val)


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e49_edge():
    """Test edge cases."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_49(2, 3)
    assert isinstance(result, dict)
    assert len(result) > 0
    first_val = next(iter(result.values()))
    assert isinstance(first_val, (int, float))
    assert math.isfinite(first_val)
