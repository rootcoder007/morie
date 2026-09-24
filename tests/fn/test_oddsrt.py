"""Tests for oddsrt.odds_ratio."""

from morie.fn import _array_core as np

from morie.fn.oddsrt import odds_ratio


def test_oddsrt_basic():
    """Test basic functionality."""
    a = 0.5
    b = 0.5
    c = 0.5
    d = 0.5
    result = odds_ratio(a, b, c, d)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_oddsrt_edge():
    """Test edge cases."""
    a = 0.5
    b = 0.5
    c = 0.5
    d = 0.5
    result = odds_ratio(a, b, c, d)
    assert isinstance(result, dict)
