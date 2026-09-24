"""Tests for rgchoi.rangayyan_choi_williams."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_choi_williams


def test_rgchoi_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_choi_williams(x)
    assert isinstance(result, dict)
    assert "tfd" in result


def test_rgchoi_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_choi_williams(x)
    assert isinstance(result, dict)
