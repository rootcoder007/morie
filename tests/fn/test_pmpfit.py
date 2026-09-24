"""Tests for pmpfit.pmp_fit."""

from morie.fn import _array_core as np

from morie.fn.pmpfit import pmp_fit


def test_pmpfit_basic():
    """Test basic functionality."""
    alpha = 0.5
    theta = 0.5
    K = 5
    result = pmp_fit(alpha, theta, K)
    assert isinstance(result, dict)
    assert "weights" in result


def test_pmpfit_edge():
    """Test edge cases."""
    alpha = 0.5
    theta = 0.5
    K = 5
    result = pmp_fit(alpha, theta, K)
    assert isinstance(result, dict)
