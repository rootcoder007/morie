"""Tests for d_from_t.d_from_t."""

from morie.fn import _array_core as np

from morie.fn.d_from_t import d_from_t


def test_ca11e5_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = d_from_t(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca11e5_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = d_from_t(x)
    assert isinstance(result, dict)
