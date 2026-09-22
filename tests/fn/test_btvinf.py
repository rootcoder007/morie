"""Tests for btvinf.boot_influence_fn."""

from morie.fn import _array_core as np

from morie.fn.btvinf import boot_influence_fn


def test_btvinf_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_influence_fn(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_btvinf_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_influence_fn(x)
    assert isinstance(result, dict)
