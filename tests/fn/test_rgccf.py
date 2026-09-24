"""Tests for rgccf.rangayyan_ccf."""

from morie.fn import _array_core as np

from morie.fn.bsacorr import rangayyan_ccf


def test_rgccf_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ccf(x, y)
    assert isinstance(result, dict)
    assert "lags" in result


def test_rgccf_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ccf(x, y)
    assert isinstance(result, dict)
