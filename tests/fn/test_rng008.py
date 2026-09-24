"""Tests for rng008.rangayyan_ch3_sample_mean_squared."""

from morie.fn import _array_core as np

from morie.fn.bsastat import rangayyan_ch3_sample_mean_squared


def test_rng008_basic():
    """Test basic functionality."""
    eta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch3_sample_mean_squared(eta)
    assert isinstance(result, dict)
    assert "mean_square" in result


def test_rng008_edge():
    """Test edge cases."""
    eta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch3_sample_mean_squared(eta)
    assert isinstance(result, dict)
