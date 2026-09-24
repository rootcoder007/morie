"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner4e8.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_8."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner4e8 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_8,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e8_basic():
    """Test basic functionality."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_8(2, 4)
    assert isinstance(result, dict)
    assert "probability" in result
    p = result["probability"]
    assert math.isfinite(p)
    assert 0.0 <= p <= 1.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e8_edge():
    """Test edge cases."""
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_8(0, 10)
    assert isinstance(result, dict)
    assert "probability" in result
    p = result["probability"]
    assert math.isfinite(p)
    assert 0.0 <= p <= 1.0
