"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner2e70.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_70."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner2e70 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_70,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e70_basic():
    """Test basic functionality with two valid probabilities."""
    p_a = 0.4
    p_b = 0.5
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_70(p_a, p_b)
    assert isinstance(result, dict)
    assert 'p_and' in result
    p_and = result['p_and']
    assert math.isfinite(p_and)
    assert 0.0 <= p_and <= 1.0
    assert result['p_a'] == p_a
    assert result['p_b'] == p_b
    assert 'ps' in result
    assert list(result['ps']) == [p_a, p_b]


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e70_edge():
    """Test edge cases with boundary probability values."""
    p_a = 0.0
    p_b = 1.0
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_70(p_a, p_b)
    assert isinstance(result, dict)
    assert 'p_and' in result
    p_and = result['p_and']
    assert math.isfinite(p_and)
    assert 0.0 <= p_and <= 1.0
    assert result['p_a'] == p_a
    assert result['p_b'] == p_b
    assert 'ps' in result
    assert list(result['ps']) == [p_a, p_b]
