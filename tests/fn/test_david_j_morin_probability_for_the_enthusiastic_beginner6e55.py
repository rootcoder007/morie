"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner6e55.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_55."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner6e55 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_55,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e55_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    y = rng.normal(0, 1, 100)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_55(x, y)
    assert isinstance(result, dict)
    assert "r" in result
    r_val = float(result["r"])
    assert math.isfinite(r_val)
    assert -1.0 <= r_val <= 1.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e55_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 20)
    y = rng.normal(0, 1, 20)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_55(x, y)
    assert isinstance(result, dict)
    assert "r" in result
    r_val = float(result["r"])
    assert math.isfinite(r_val)
    assert -1.0 <= r_val <= 1.0
