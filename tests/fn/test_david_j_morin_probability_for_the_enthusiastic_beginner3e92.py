"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner3e92.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_92."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner3e92 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_92,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e92_basic():
    """Test basic functionality."""
    sigma = 2.5
    N = 50
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_92(sigma, N)
    assert isinstance(result, dict)
    assert "var_mean" in result
    assert math.isfinite(float(result["var_mean"]))
    assert float(result["var_mean"]) >= 0.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e92_edge():
    """Test edge cases."""
    sigma = 1.0
    N = 1
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_92(sigma, N)
    assert isinstance(result, dict)
    assert "var_mean" in result
    assert math.isfinite(float(result["var_mean"]))
    assert float(result["var_mean"]) >= 0.0
