"""Tests for rdksrn.sharp_rdd."""

from morie.fn import _array_core as np

from morie.fn.rdksrn import sharp_rdd


def test_rdksrn_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = sharp_rdd(y, x)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_rdksrn_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = sharp_rdd(y, x)
    assert isinstance(result, dict)
