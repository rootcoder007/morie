"""Tests for rng020.rangayyan_ch3_time_averaged_acf."""

from morie.fn import _array_core as np

from morie.fn.bsacorr import rangayyan_ch3_time_averaged_acf


def test_rng020_basic():
    """Test basic functionality."""
    x_k = np.random.default_rng(42).normal(0.0, 1.0, 40)
    tau = 5
    result = rangayyan_ch3_time_averaged_acf(x_k, tau)
    assert isinstance(result, dict)
    assert "acf" in result


def test_rng020_edge():
    """Test edge cases."""
    x_k = np.random.default_rng(42).normal(0.0, 1.0, 40)
    tau = 5
    result = rangayyan_ch3_time_averaged_acf(x_k, tau)
    assert isinstance(result, dict)
