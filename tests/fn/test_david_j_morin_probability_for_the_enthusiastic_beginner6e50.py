"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner6e50.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_50."""

import pytest

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner6e50 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_50,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e50_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    y = rng.normal(0, 1, n)
    x = rng.normal(0, 1, n)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_50(x, y)
    assert isinstance(result, dict)
    assert "C" in result
    assert "D" in result
    assert "S" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e50_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 10
    y = rng.normal(0, 1, n)
    x = rng.normal(0, 1, n)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_50(x, y)
    assert isinstance(result, dict)
    assert len(result) >= 3
