"""Tests for excess_score_factor.excess_score_factor."""

import math

from morie.fn import _array_core as np

from morie.fn.excess_score_factor import (
    excess_score_factor,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e81_basic():
    """Test basic functionality."""
    r = 0.5
    result = excess_score_factor(r)
    assert isinstance(result, dict)
    assert "factor" in result
    assert math.isfinite(result["factor"])
    assert result["factor"] >= 0.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e81_edge():
    """Test edge cases."""
    r = 0.0
    result = excess_score_factor(r)
    assert isinstance(result, dict)
    assert "factor" in result
    assert math.isfinite(result["factor"])
    assert result["factor"] >= 0.0
