"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner3e73.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_73."""

from morie.fn import _array_core as np
from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner3e73 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_73,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e73_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_73(x)
    assert isinstance(result, dict)
    assert "sample_variance" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e73_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_73(x)
    assert isinstance(result, dict)


def test_alias_warns_and_returns_the_sample_variance():
    import pytest

    x = [2.0, 4.5, 3.0, 7.5, 1.0]
    m = sum(x) / 5
    with pytest.warns(DeprecationWarning):
        r = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_73(x)
    assert r["sample_variance"] == pytest.approx(sum((v - m) ** 2 for v in x) / 4, rel=1e-13)
