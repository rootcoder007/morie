"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner7e35.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_35."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner7e35 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_35,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner7e35_basic():
    """Test basic functionality."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_35(
        60, 100, 0.1
    )
    assert isinstance(result, dict)
    assert len(result) > 0
    # The function returns a RichResult with at least one numeric, finite value.
    numeric_values = [
        v for v in result.values()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    assert len(numeric_values) > 0
    assert all(math.isfinite(v) for v in numeric_values)


def test_david_j_morin_probability_for_the_enthusiastic_beginner7e35_edge():
    """Test edge cases."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_35(
        50, 100, 0.0
    )
    assert isinstance(result, dict)
    assert len(result) > 0
    numeric_values = [
        v for v in result.values()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    assert len(numeric_values) > 0
    assert all(math.isfinite(v) for v in numeric_values)
