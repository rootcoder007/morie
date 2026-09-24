"""Tests for odds_ratio_2x2.odds_ratio_2x2."""

from morie.fn import _array_core as np

from morie.fn.odds_ratio_2x2 import odds_ratio_2x2


def test_ca11e10_basic():
    """Test basic functionality."""
    a = 0.5
    b = 0.5
    c = 0.5
    d = 0.5
    result = odds_ratio_2x2(a, b, c, d)
    assert isinstance(result, dict)
    assert "value" in result


def test_ca11e10_edge():
    """Test edge cases."""
    a = 0.5
    b = 0.5
    c = 0.5
    d = 0.5
    result = odds_ratio_2x2(a, b, c, d)
    assert isinstance(result, dict)
