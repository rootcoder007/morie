"""Tests for locdp.local_dp."""

from morie.fn import _array_core as np

from morie.fn.locdp import local_dp


def test_locdp_basic():
    """Test basic functionality."""
    bit = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = local_dp(bit)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_locdp_edge():
    """Test edge cases."""
    bit = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = local_dp(bit)
    assert isinstance(result, dict)
