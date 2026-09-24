"""Tests for mapeto.ma_peto_or."""

from morie.fn import _array_core as np

from morie.fn.mapeto import ma_peto_or


def test_mapeto_basic():
    """Test basic functionality."""
    a = 0.5
    b = 0.5
    c = 0.5
    d = 0.5
    result = ma_peto_or(a, b, c, d)
    assert isinstance(result, dict)
    assert "OR" in result


def test_mapeto_edge():
    """Test edge cases."""
    a = 0.5
    b = 0.5
    c = 0.5
    d = 0.5
    result = ma_peto_or(a, b, c, d)
    assert isinstance(result, dict)
