"""Tests for rmsoptm.rmsprop."""

from morie.fn import _array_core as np

from morie.fn.rmsoptm import rmsprop


def test_rmsoptm_basic():
    """Test basic functionality."""
    g = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rmsprop(g)
    assert isinstance(result, dict)
    assert "update" in result


def test_rmsoptm_edge():
    """Test edge cases."""
    g = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rmsprop(g)
    assert isinstance(result, dict)
