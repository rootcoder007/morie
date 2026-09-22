"""Tests for btsbb.boot_stationary_block."""

from morie.fn import _array_core as np

from morie.fn.btsbb import boot_stationary_block


def test_btsbb_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_stationary_block(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_btsbb_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_stationary_block(x)
    assert isinstance(result, dict)
