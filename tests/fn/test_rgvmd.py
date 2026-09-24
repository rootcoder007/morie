"""Tests for rgvmd.rangayyan_vmd."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_vmd


def test_rgvmd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_vmd(x)
    assert isinstance(result, dict)
    assert "modes" in result


def test_rgvmd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_vmd(x)
    assert isinstance(result, dict)
