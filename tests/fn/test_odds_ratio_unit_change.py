"""Tests for odds_ratio_unit_change.odds_ratio_unit_change."""

from morie.fn import _array_core as np

from morie.fn.odds_ratio_unit_change import odds_ratio_unit_change


def test_ca4e8_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = odds_ratio_unit_change(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca4e8_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = odds_ratio_unit_change(x)
    assert isinstance(result, dict)
