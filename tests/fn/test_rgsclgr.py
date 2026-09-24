"""Tests for rgsclgr.rangayyan_scalogram."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_scalogram


def test_rgsclgr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_scalogram(x)
    assert isinstance(result, dict)
    assert "scalogram" in result


def test_rgsclgr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_scalogram(x)
    assert isinstance(result, dict)
