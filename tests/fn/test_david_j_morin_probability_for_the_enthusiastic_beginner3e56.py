"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner3e56.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_56."""

import math

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner3e56 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_56,
)


def _extract_sd(result):
    """Extract the standard deviation value from the result's string representation."""
    text = repr(result)
    lines = [line for line in text.split("\n") if line.strip()]
    last_line = lines[-1]
    parts = last_line.split()
    return float(parts[-1])


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e56_basic():
    """Test basic functionality."""
    n = 10000
    p = 1.0 / 6.0
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_56(
        n=n, p=p
    )
    value = _extract_sd(result)
    assert math.isfinite(value)
    assert value >= 0
    # binomial SD <= sqrt(n / 4)
    assert value <= math.sqrt(n / 4) + 1e-9


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e56_edge():
    """Test edge cases."""
    n = 10
    p = 0.5
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_56(
        n=n, p=p
    )
    value = _extract_sd(result)
    assert math.isfinite(value)
    assert value >= 0
    # binomial SD <= sqrt(n / 4)
    assert value <= math.sqrt(n / 4) + 1e-9
