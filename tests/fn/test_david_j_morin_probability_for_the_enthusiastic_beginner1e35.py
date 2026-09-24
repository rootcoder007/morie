"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner1e35.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_35."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner1e35 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_35,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e35_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    ns = [int(x) for x in rng.integers(0, 10, 5)]
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_35(ns)
    assert isinstance(result, dict)
    assert any(
        isinstance(v, (int, float)) and math.isfinite(v) and v >= 1
        for v in result.values()
    )


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e35_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    ns = [int(x) for x in rng.integers(0, 10, 5)]
    N = sum(ns)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_35(ns, N)
    assert isinstance(result, dict)
    assert any(
        isinstance(v, (int, float)) and math.isfinite(v) and v >= 1
        for v in result.values()
    )
