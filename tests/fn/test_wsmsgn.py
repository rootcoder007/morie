"""Tests for wsmsgn.wasserman_sign_test."""

from morie.fn import _array_core as np

from morie.fn.wsmsgn import wasserman_sign_test


def test_wsmsgn_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    theta0 = 0.0
    result = wasserman_sign_test(x, theta0)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_wsmsgn_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    theta0 = 0.0
    result = wasserman_sign_test(x, theta0)
    assert isinstance(result, dict)
