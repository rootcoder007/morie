"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner3e54.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_54."""

from morie.fn import _array_core as np
from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner3e54 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_54,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e54_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_54(x)
    assert isinstance(result, dict)
    assert "mean" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e54_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_54(x)
    assert isinstance(result, dict)


def test_alias_warns_and_returns_the_sample_mean():
    import pytest

    x = [2.0, 4.5, 3.0, 7.5, 1.0]
    with pytest.warns(DeprecationWarning):
        r = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_54(x)
    assert r["mean"] == pytest.approx(3.6, rel=1e-15)
