"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner4e85.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_85."""

import math

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner4e85 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_85,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e85_basic():
    """Test basic functionality."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_85(2.0)
    # expmom2 returns exponential moments (mean, second moment, variance) for a scalar tau
    if isinstance(result, dict):
        assert len(result) >= 1
        for v in result.values():
            assert math.isfinite(float(v))
    else:
        values = list(result)
        assert len(values) == 3
        for v in values:
            assert math.isfinite(float(v))


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e85_edge():
    """Test edge cases."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_85(0.5)
    if isinstance(result, dict):
        assert len(result) >= 1
        for v in result.values():
            assert math.isfinite(float(v))
    else:
        values = list(result)
        assert len(values) == 3
        for v in values:
            assert math.isfinite(float(v))
