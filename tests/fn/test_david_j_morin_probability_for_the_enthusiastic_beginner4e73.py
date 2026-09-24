"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner4e73.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_73."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner4e73 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_73,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e73_basic():
    """Test basic functionality."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_73(
        k=5, n=10, p=0.5, N=20
    )
    assert isinstance(result, dict)
    assert "hypergeometric" in result.payload
    assert "binomial" in result.payload
    assert "abs_error" in result.payload
    assert math.isfinite(result.payload["hypergeometric"])
    assert math.isfinite(result.payload["binomial"])
    assert math.isfinite(result.payload["abs_error"])
    assert 0.0 <= result.payload["hypergeometric"] <= 1.0
    assert 0.0 <= result.payload["binomial"] <= 1.0
    assert result.payload["abs_error"] >= 0.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e73_edge():
    """Test edge cases."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_73(
        k=1, n=1, p=0.5, N=2
    )
    assert isinstance(result, dict)
    assert "hypergeometric" in result.payload
    assert "binomial" in result.payload
    assert "abs_error" in result.payload
    assert math.isfinite(result.payload["hypergeometric"])
    assert math.isfinite(result.payload["binomial"])
    assert math.isfinite(result.payload["abs_error"])
    assert 0.0 <= result.payload["hypergeometric"] <= 1.0
    assert 0.0 <= result.payload["binomial"] <= 1.0
    assert result.payload["abs_error"] >= 0.0
