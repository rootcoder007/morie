"""Tests for arimax.arimax."""

from morie.fn import _array_core as np

from morie.fn.arimax import arimax


def test_arimax_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = arimax(y, X)
    assert isinstance(result, dict)
    assert "beta" in result
def test_arimax_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = arimax(y, X)
    assert isinstance(result, dict)
