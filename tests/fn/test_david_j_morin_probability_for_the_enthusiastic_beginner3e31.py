"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner3e31.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_31."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner3e31 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_31,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e31_basic():
    """Test basic functionality."""
    variances = [0.5, 1.0, 2.0, 0.25]
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_31(variances)
    assert "var_sum" in result.payload
    var_sum = result["var_sum"]
    assert math.isfinite(var_sum)
    assert var_sum >= 0
    assert "partial_sums" in result.payload
    assert len(result["partial_sums"]) == len(variances)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e31_edge():
    """Test edge cases."""
    variances = [0.0, 1.5]
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_31(variances)
    assert "var_sum" in result.payload
    var_sum = result["var_sum"]
    assert math.isfinite(var_sum)
    assert var_sum >= 0
    assert "partial_sums" in result.payload
    assert len(result["partial_sums"]) == len(variances)
