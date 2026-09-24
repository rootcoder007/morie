"""Tests for se_log_rr.se_log_rr."""

from morie.fn import _array_core as np

from morie.fn.se_log_rr import se_log_rr


def test_ca11e9_basic():
    """Test basic functionality."""
    p1 = 0.5
    p2 = 0.5
    n1 = 0.5
    n2 = 0.5
    result = se_log_rr(p1, p2, n1, n2)
    assert isinstance(result, dict)
    assert "value" in result or "value" in result


def test_ca11e9_edge():
    """Test edge cases."""
    p1 = 0.5
    p2 = 0.5
    n1 = 0.5
    n2 = 0.5
    result = se_log_rr(p1, p2, n1, n2)
    assert isinstance(result, dict)
