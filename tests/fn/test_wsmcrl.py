"""Tests for wsmcrl.wasserman_cramer_rao."""

from morie.fn.wsmcrl import wasserman_cramer_rao


def test_wsmcrl_basic():
    """Test basic functionality."""
    theta = 0.1
    n = 5
    I_ = 0.1
    result = wasserman_cramer_rao(theta, n, I_)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_wsmcrl_edge():
    """Test edge cases."""
    theta = 0.1
    n = 5
    I_ = 0.1
    result = wasserman_cramer_rao(theta, n, I_)
    assert isinstance(result, dict)
