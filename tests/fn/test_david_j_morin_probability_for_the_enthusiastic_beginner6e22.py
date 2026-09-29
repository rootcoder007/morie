"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner6e22.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_22."""

from morie.fn import _array_core as np
from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner6e22 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_22,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e22_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_22(x)
    assert isinstance(result, dict)
    assert "best_prediction" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e22_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_22(x)
    assert isinstance(result, dict)


def test_alias_warns_and_returns_the_best_constant():
    import pytest

    y = [2.0, 4.5, 3.0, 7.5, 1.0]
    with pytest.warns(DeprecationWarning):
        r = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_22(y)
    assert r["best_prediction"] == pytest.approx(3.6, rel=1e-15)
    assert r["mse"] == pytest.approx(sum((v - 3.6) ** 2 for v in y) / 5, rel=1e-13)
