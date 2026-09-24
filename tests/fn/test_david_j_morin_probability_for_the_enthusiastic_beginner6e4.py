"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner6e4.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_4."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner6e4 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_4,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e4_basic():
    """Test basic functionality."""
    m = 1
    mu_x = 0
    mu_z = 1
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_4(m, mu_x, mu_z)
    assert isinstance(result, dict)
    assert "mu_y" in result
    assert isinstance(result["mu_y"], (int, float))
    assert math.isfinite(result["mu_y"])


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e4_edge():
    """Test edge cases."""
    m = 3
    mu_x = 0
    mu_z = 1
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_4(m, mu_x, mu_z)
    assert isinstance(result, dict)
    assert "mu_y" in result
    assert isinstance(result["mu_y"], (int, float))
    assert math.isfinite(result["mu_y"])
