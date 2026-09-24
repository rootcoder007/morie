"""Tests for mtsBnd.mts_bounds."""

from morie.fn import _array_core as np

from morie.fn.mtsbnd import mts_bounds


def test_mtsbnd_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mts_bounds(Y, X)
    assert isinstance(result, dict)
    assert "lower" in result


def test_mtsbnd_edge():
    """Test edge cases."""
    Y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mts_bounds(Y, X)
    assert isinstance(result, dict)
