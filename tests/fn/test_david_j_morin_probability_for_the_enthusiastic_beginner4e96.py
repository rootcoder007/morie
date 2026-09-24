"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner4e96.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_96."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner4e96 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_96,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e96_basic():
    """Test basic functionality."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_96(10, 0.5)
    # The function returns a dict-like RichResult containing the binomial peak mode and probability
    assert isinstance(result, dict)
    # Verify that the result contains a probability (a number between 0 and 1)
    prob = None
    for value in result.values():
        if isinstance(value, (int, float)) and 0 <= value <= 1:
            prob = value
            break
    assert prob is not None
    assert math.isfinite(prob)
    # Verify that the result contains a mode (an integer)
    mode = None
    for value in result.values():
        if isinstance(value, int):
            mode = value
            break
    assert mode is not None


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e96_edge():
    """Test edge cases."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_96(1, 0.0)
    assert isinstance(result, dict)
    prob = None
    for value in result.values():
        if isinstance(value, (int, float)) and 0 <= value <= 1:
            prob = value
            break
    assert prob is not None
    assert math.isfinite(prob)
    mode = None
    for value in result.values():
        if isinstance(value, int):
            mode = value
            break
    assert mode is not None
