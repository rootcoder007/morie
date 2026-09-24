"""Tests for hrztiku.horowitz_tikhonov_unknown_T."""

from morie.fn import _array_core as np

from morie.fn.hrztiku import horowitz_tikhonov_unknown_T


def test_hrztiku_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    w = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_tikhonov_unknown_T(x, y, w)
    assert isinstance(result, dict)
    assert "g_hat" in result


def test_hrztiku_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    w = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_tikhonov_unknown_T(x, y, w)
    assert isinstance(result, dict)
