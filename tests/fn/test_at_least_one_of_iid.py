"""Tests for at_least_one_of_iid.at_least_one_of_iid."""

import math

from morie.fn import _array_core as np

from morie.fn.at_least_one_of_iid import (
    at_least_one_of_iid,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e96_basic():
    """Test basic functionality."""
    p = 0.5
    k = 3
    result = at_least_one_of_iid(p, k)
    assert isinstance(result, dict)
    assert "p_at_least_one" in result
    value = result["p_at_least_one"]
    assert math.isfinite(value)
    assert 0.0 <= value <= 1.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e96_edge():
    """Test edge cases."""
    p = 0.2
    k = 5
    result = at_least_one_of_iid(p, k)
    assert isinstance(result, dict)
    assert "p_at_least_one" in result
    assert "p" in result
    assert "k" in result
    value = result["p_at_least_one"]
    assert math.isfinite(value)
    assert 0.0 <= value <= 1.0
    assert result["p"] == float(p)
    assert result["k"] == int(k)
