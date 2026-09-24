"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner1e56.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_56."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner1e56 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_56,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e56_basic():
    """Test basic functionality."""
    n, k = 10, 3
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_56(n, k)
    assert isinstance(result, dict)
    assert len(result) > 0
    for value in result.values():
        assert isinstance(value, (int, float))
        assert math.isfinite(value)
        assert value > 0


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e56_edge():
    """Test edge cases."""
    n, k = 5, 1
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_56(n, k)
    assert isinstance(result, dict)
    assert len(result) > 0
    for value in result.values():
        assert isinstance(value, (int, float))
        assert math.isfinite(value)
        assert value > 0
