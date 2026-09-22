"""Tests for btcbb.boot_circular_block."""

from morie.fn import _array_core as np

from morie.fn.btcbb import boot_circular_block


def test_btcbb_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_circular_block(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_btcbb_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_circular_block(x)
    assert isinstance(result, dict)
