"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner3e30.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_30."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner3e30 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_30,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e30_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    variances = [v * v for v in x]
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_30(variances)
    assert isinstance(result, dict)
    assert "var_sum" in result
    assert "partial_sums" in result
    assert math.isfinite(result["var_sum"])


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e30_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    variances = [abs(v) for v in x]
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_30(variances)
    assert isinstance(result, dict)
    assert "var_sum" in result
    assert "partial_sums" in result
    assert math.isfinite(result["var_sum"])
