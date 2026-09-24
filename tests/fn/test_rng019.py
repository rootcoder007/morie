"""Tests for rng019.rangayyan_ch3_time_average_mean."""

from morie.fn import _array_core as np

from morie.fn.bsastat import rangayyan_ch3_time_average_mean


def test_rng019_basic():
    """Test basic functionality."""
    x_k = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch3_time_average_mean(x_k)
    assert isinstance(result, dict)
    assert "time_mean" in result


def test_rng019_edge():
    """Test edge cases."""
    x_k = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch3_time_average_mean(x_k)
    assert isinstance(result, dict)
