"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner5e4.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_4."""

import math

import pytest

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner5e4 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_4,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner5e4_basic():
    """Test basic functionality."""
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_4(0.5, 10)
    assert isinstance(result, dict)
    for key in ("approx", "exact", "rel_error"):
        assert key in result
        assert math.isfinite(float(result[key]))


def test_david_j_morin_probability_for_the_enthusiastic_beginner5e4_edge():
    """Test edge cases."""
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_4(0.0, 1)
    assert isinstance(result, dict)
    for key in ("approx", "exact", "rel_error"):
        assert key in result
        assert math.isfinite(float(result[key]))
