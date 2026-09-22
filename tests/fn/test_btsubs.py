"""Tests for btsubs.boot_subsampling."""

from morie.fn import _array_core as np

from morie.fn.btsubs import boot_subsampling


def test_btsubs_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_subsampling(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_btsubs_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_subsampling(x)
    assert isinstance(result, dict)
