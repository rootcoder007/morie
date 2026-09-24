"""Tests for sd_fair_coin_avg.sd_fair_coin_avg."""

from morie.fn import _array_core as np

from morie.fn.sd_fair_coin_avg import (
    sd_fair_coin_avg,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e52_basic():
    """Test basic functionality."""
    n = 5
    result = sd_fair_coin_avg(n)
    assert isinstance(result, dict)
    assert "n" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e52_edge():
    """Test edge cases."""
    n = 5
    result = sd_fair_coin_avg(n)
    assert isinstance(result, dict)
