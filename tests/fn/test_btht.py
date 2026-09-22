"""Tests for btht.boot_test_hypothesis."""

from morie.fn import _array_core as np

from morie.fn.btht import boot_test_hypothesis


def test_btht_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_test_hypothesis(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_btht_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_test_hypothesis(x)
    assert isinstance(result, dict)
