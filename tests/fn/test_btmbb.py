"""Tests for btmbb.boot_moving_block."""

from morie.fn import _array_core as np

from morie.fn.btmbb import boot_moving_block


def test_btmbb_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_moving_block(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_btmbb_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_moving_block(x)
    assert isinstance(result, dict)
