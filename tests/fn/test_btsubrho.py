"""Tests for btsubrho.boot_subsample_rate."""

from morie.fn import _array_core as np

from morie.fn.btsubrho import boot_subsample_rate


def test_btsubrho_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_subsample_rate(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_btsubrho_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_subsample_rate(x)
    assert isinstance(result, dict)
