"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner4e34.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_34."""

import math

from morie.fn import _array_core as np
from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner4e34 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_34,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e34_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    k = int(rng.integers(0, 5))
    n = int(rng.integers(5, 20))
    a = float(rng.uniform(0.1, 0.9))
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_34(k, n, a)
    assert isinstance(result, dict)
    assert "binomial" in result
    assert "poisson" in result
    assert "abs_error" in result
    assert math.isfinite(result["binomial"])
    assert 0 <= result["binomial"] <= 1
    assert math.isfinite(result["poisson"])
    assert 0 <= result["poisson"] <= 1
    assert math.isfinite(result["abs_error"])
    assert result["abs_error"] >= 0


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e34_edge():
    """Test edge cases."""
    # Edge case: k = 0 successes.
    k = 0
    n = 10
    a = 0.3
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_34(k, n, a)
    assert isinstance(result, dict)
    assert "binomial" in result
    assert "poisson" in result
    assert "abs_error" in result
    assert math.isfinite(result["binomial"])
    assert 0 <= result["binomial"] <= 1
    assert math.isfinite(result["poisson"])
    assert 0 <= result["poisson"] <= 1
    assert math.isfinite(result["abs_error"])
    assert result["abs_error"] >= 0
