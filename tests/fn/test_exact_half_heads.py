"""Tests for exact_half_heads.exact_half_heads."""

import math

from morie.fn.exact_half_heads import (
    exact_half_heads,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e65_basic():
    """Test basic functionality."""
    result = exact_half_heads(5)
    assert isinstance(result, dict)
    assert "n" in result
    assert "probability" in result
    assert result["n"] == 5
    assert math.isfinite(result["probability"])
    assert 0.0 <= result["probability"] <= 1.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e65_edge():
    """Test edge cases."""
    result = exact_half_heads(1)
    assert isinstance(result, dict)
    assert "n" in result
    assert "probability" in result
    assert result["n"] == 1
    assert math.isfinite(result["probability"])
    assert 0.0 <= result["probability"] <= 1.0
