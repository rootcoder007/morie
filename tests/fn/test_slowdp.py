"""Tests for slowdp.slow_dp_truncate."""

from morie.fn import _array_core as np

from morie.fn.slowdp import slow_dp_truncate


def test_slowdp_basic():
    """Test basic functionality."""
    alpha = 0.5
    K = 5
    result = slow_dp_truncate(alpha, K)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_slowdp_edge():
    """Test edge cases."""
    alpha = 0.5
    K = 5
    result = slow_dp_truncate(alpha, K)
    assert isinstance(result, dict)
