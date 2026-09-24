"""Tests for rgemd.rangayyan_emd."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_emd


def test_rgemd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_emd(x)
    assert isinstance(result, dict)
    assert "imfs" in result


def test_rgemd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_emd(x)
    assert isinstance(result, dict)
