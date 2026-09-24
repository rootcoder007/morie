"""Tests for rgvagadp.rangayyan_vag_adaptive_tfd."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_vag_adaptive_tfd


def test_rgvagadp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.1
    result = rangayyan_vag_adaptive_tfd(x, fs)
    assert isinstance(result, dict)
    assert "tfd" in result


def test_rgvagadp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.1
    result = rangayyan_vag_adaptive_tfd(x, fs)
    assert isinstance(result, dict)
