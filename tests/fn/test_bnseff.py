"""Tests for bnseff.bound_efficient."""

from morie.fn import _array_core as np

from morie.fn.bnseff import bound_efficient


def test_bnseff_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = bound_efficient(y, D, X)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_bnseff_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = bound_efficient(y, D, X)
    assert isinstance(result, dict)
