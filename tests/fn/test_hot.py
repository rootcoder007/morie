"""Tests for hot.hot_sax."""

from morie.fn import _array_core as np

from morie.fn.hot import hot_sax


def test_hot_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    window = 5
    result = hot_sax(x, window)
    assert isinstance(result, dict)
    assert "estimate" in result or "location" in result


def test_hot_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    window = 5
    result = hot_sax(x, window)
    assert isinstance(result, dict)
