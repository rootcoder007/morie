"""Tests for se_log_or.se_log_or."""

from morie.fn import _array_core as np

from morie.fn.se_log_or import se_log_or


def test_ca11e11_basic():
    """Test basic functionality."""
    a = 0.5
    b = 0.5
    c = 0.5
    d = 0.5
    result = se_log_or(a, b, c, d)
    assert isinstance(result, dict)
    assert "value" in result


def test_ca11e11_edge():
    """Test edge cases."""
    a = 0.5
    b = 0.5
    c = 0.5
    d = 0.5
    result = se_log_or(a, b, c, d)
    assert isinstance(result, dict)
