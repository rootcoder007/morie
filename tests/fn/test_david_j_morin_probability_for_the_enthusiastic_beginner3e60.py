"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner3e60.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_60."""

from morie.fn import _array_core as np
from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner3e60 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_60,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e60_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_60(x)
    assert isinstance(result, dict)
    assert "variance" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e60_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_60(x)
    assert isinstance(result, dict)


def test_alias_warns_and_returns_the_population_variance():
    import pytest

    x = [2.0, 4.5, 3.0, 7.5, 1.0]
    m = sum(x) / 5
    with pytest.warns(DeprecationWarning):
        r = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_60(x)
    assert r["variance"] == pytest.approx(sum((v - m) ** 2 for v in x) / 5, rel=1e-13)
