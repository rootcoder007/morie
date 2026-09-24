"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner3e59.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_59."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner3e59 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_59,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e59_basic():
    """Test basic functionality."""
    values = [0.0, 1.0, 2.0, 3.0, 4.0]
    probs = [0.1, 0.2, 0.3, 0.3, 0.1]
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_59(values, probs)
    assert isinstance(result, dict)
    assert "variance" in result
    assert math.isfinite(result["variance"])
    # Verify against direct computation
    mean = sum(v * p for v, p in zip(values, probs))
    expected_var = sum(p * (v - mean) ** 2 for v, p in zip(values, probs))
    assert abs(result["variance"] - expected_var) < 1e-9


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e59_edge():
    """Test edge cases."""
    values = [0.0, 1.0]
    probs = [0.5, 0.5]
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_59(values, probs)
    assert isinstance(result, dict)
    assert "variance" in result
    assert math.isfinite(result["variance"])
    # Verify against direct computation
    mean = sum(v * p for v, p in zip(values, probs))
    expected_var = sum(p * (v - mean) ** 2 for v, p in zip(values, probs))
    assert abs(result["variance"] - expected_var) < 1e-9
