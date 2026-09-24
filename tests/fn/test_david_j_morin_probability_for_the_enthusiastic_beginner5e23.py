"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner5e23.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_23."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner5e23 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_23,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner5e23_basic():
    """Test basic functionality."""
    k = 3
    a = 2.0
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_23(k, a)
    assert isinstance(result, dict)
    assert len(result) >= 1
    value = next(iter(result.values()))
    assert isinstance(value, (int, float))
    assert math.isfinite(value)
    assert 0 <= value <= 1


def test_david_j_morin_probability_for_the_enthusiastic_beginner5e23_edge():
    """Test edge cases."""
    k = 0
    a = 1.0
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_23(k, a)
    assert isinstance(result, dict)
    assert len(result) >= 1
    value = next(iter(result.values()))
    assert isinstance(value, (int, float))
    assert math.isfinite(value)
    assert 0 <= value <= 1
