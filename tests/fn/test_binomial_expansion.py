"""Tests for binomial_expansion.binomial_expansion."""
import math

from morie.fn.binomial_expansion import (
    binomial_expansion,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e21_basic():
    """Test basic functionality."""
    result = binomial_expansion(2, 3, 4)
    assert isinstance(result, dict)
    # (2+3)^4 = 625
    assert math.isfinite(result.payload["direct"])
    assert result.payload["sum"] == result.payload["direct"]
    assert result.payload["max_abs_error"] == 0.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e21_edge():
    """Test edge cases."""
    result = binomial_expansion(5, 7, 1)
    assert isinstance(result, dict)
    # (5+7)^1 = 12
    assert result.payload["sum"] == 12.0
    assert result.payload["direct"] == 12.0
    assert result.payload["max_abs_error"] == 0.0
