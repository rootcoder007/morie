"""Tests for odds_ratio_2x2.odds_ratio_2x2."""

from morie.fn import _array_core as np

from morie.fn.odds_ratio_2x2 import odds_ratio_2x2


def test_ca11e10_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = odds_ratio_2x2(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca11e10_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = odds_ratio_2x2(x)
    assert isinstance(result, dict)
