"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner4e6.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_6."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner4e6 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_6,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e6_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 20
    p = 0.4
    k = int(rng.integers(0, n + 1))
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_6(k, n, p)
    assert isinstance(result, dict)
    assert set(["k", "n", "p", "probability"]).issubset(set(result.keys()))
    prob = result["probability"]
    assert math.isfinite(float(prob))
    assert 0.0 <= float(prob) <= 1.0
    assert int(result["k"]) == k
    assert int(result["n"]) == n
    assert float(result["p"]) == p


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e6_edge():
    """Test edge cases."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_6(0, 10, 0.0)
    assert isinstance(result, dict)
    assert set(["k", "n", "p", "probability"]).issubset(set(result.keys()))
    prob = float(result["probability"])
    assert math.isfinite(prob)
    assert 0.0 <= prob <= 1.0
