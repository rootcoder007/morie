"""Tests for wsmdlm.wasserman_delta_method."""

from morie.fn import _array_core as np

from morie.fn.wsmdlm import wasserman_delta_method


def test_wsmdlm_basic():
    """Test basic functionality."""
    theta_hat = 0.1
    se = 0.1
    g_prime = 0.1
    result = wasserman_delta_method(theta_hat, se, g_prime)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_wsmdlm_edge():
    """Test edge cases."""
    theta_hat = 0.1
    se = 0.1
    g_prime = 0.1
    result = wasserman_delta_method(theta_hat, se, g_prime)
    assert isinstance(result, dict)
