"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner1e71.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_71."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner1e71 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_71,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e71_basic():
    """Test basic functionality."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_71(1.0, 100)
    assert isinstance(result, dict)
    assert 'sd_sum' in result
    assert isinstance(result['sd_sum'], (int, float))
    assert math.isfinite(result['sd_sum'])


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e71_edge():
    """Test edge cases."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_71(0.5, 1)
    assert isinstance(result, dict)
    assert 'sd_sum' in result
    assert isinstance(result['sd_sum'], (int, float))
    assert math.isfinite(result['sd_sum'])
