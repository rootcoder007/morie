"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner3e35.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_35."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner3e35 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_35,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e35_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    values = [0.0, 1.0, 2.0, 3.0, 4.0]
    probs = [0.1, 0.2, 0.3, 0.3, 0.1]
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_35(values, probs)
    assert isinstance(result, dict)
    assert "variance" in result
    assert math.isfinite(float(result["variance"]))


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e35_edge():
    """Test edge cases."""
    values = [1.0, 2.0]
    probs = [0.5, 0.5]
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_35(values, probs)
    assert isinstance(result, dict)
    assert "variance" in result
    assert math.isfinite(float(result["variance"]))
