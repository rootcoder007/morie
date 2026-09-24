"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner4e35.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_35."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner4e35 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_35,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e35_basic():
    """Test basic functionality."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_35(5, 10, 0.5)
    assert isinstance(result, dict)
    assert "binomial" in result
    assert "poisson" in result
    assert "abs_error" in result
    binomial = result["binomial"]
    poisson = result["poisson"]
    assert math.isfinite(binomial)
    assert 0 <= binomial <= 1
    assert math.isfinite(poisson)
    assert 0 <= poisson <= 1


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e35_edge():
    """Test edge cases."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_35(0, 1, 0.5)
    assert isinstance(result, dict)
    assert "binomial" in result
    assert "poisson" in result
    assert "abs_error" in result
    binomial = result["binomial"]
    poisson = result["poisson"]
    assert math.isfinite(binomial)
    assert 0 <= binomial <= 1
    assert math.isfinite(poisson)
    assert 0 <= poisson <= 1
