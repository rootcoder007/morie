"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner3e57.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_57."""

import math

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner3e57 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_57,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e57_basic():
    """Test basic functionality with the textbook default values."""
    n = 10000
    p = 1.0 / 6.0
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_57(n=n, p=p)
    assert isinstance(result, dict)
    assert math.isfinite(result["sd_avg"])
    assert math.isclose(result["sd_avg"], math.sqrt(p * (1 - p) / n))


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e57_edge():
    """Test edge case: p = 0.5 maximises the binomial standard deviation."""
    n = 50
    p = 0.5
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_57(n=n, p=p)
    assert isinstance(result, dict)
    assert math.isfinite(result["sd_avg"])
    assert math.isclose(result["sd_avg"], math.sqrt(p * (1 - p) / n))
