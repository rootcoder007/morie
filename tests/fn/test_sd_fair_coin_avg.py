"""Tests for sd_fair_coin_avg.sd_fair_coin_avg."""

from morie.fn import _array_core as np

from morie.fn.sd_fair_coin_avg import (
    sd_fair_coin_avg,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e52_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = sd_fair_coin_avg(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e52_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = sd_fair_coin_avg(x)
    assert isinstance(result, dict)
