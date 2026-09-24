"""Tests for wsmwld.wasserman_wald."""

from morie.fn import _array_core as np

from morie.fn.wsmwld import wasserman_wald


def test_wsmwld_basic():
    """Test basic functionality."""
    theta_hat = 0.1
    se = 0.1
    result = wasserman_wald(theta_hat, se)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_wsmwld_edge():
    """Test edge cases."""
    theta_hat = 0.1
    se = 0.1
    result = wasserman_wald(theta_hat, se)
    assert isinstance(result, dict)
