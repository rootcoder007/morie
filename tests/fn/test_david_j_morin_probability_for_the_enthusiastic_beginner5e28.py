"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner5e28.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_28."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner5e28 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_28,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner5e28_basic():
    """Test basic functionality."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_28(40.0)
    assert isinstance(result, dict)
    assert len(result) > 0
    # Find a count-like value (non-negative finite number)
    count_value = None
    for v in result.values():
        if isinstance(v, (int, float)) and math.isfinite(v) and v >= 0:
            count_value = v
            break
    assert count_value is not None
    assert math.isfinite(count_value)


def test_david_j_morin_probability_for_the_enthusiastic_beginner5e28_edge():
    """Test edge cases."""
    # Use a smaller n_reps for faster edge-case test
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_28(
        30.0, n_reps=1000, mu=35.0, sigma=5.4
    )
    assert isinstance(result, dict)
    assert len(result) > 0
    # Find a count-like value (non-negative finite number)
    count_value = None
    for v in result.values():
        if isinstance(v, (int, float)) and math.isfinite(v) and v >= 0:
            count_value = v
            break
    assert count_value is not None
    assert math.isfinite(count_value)
