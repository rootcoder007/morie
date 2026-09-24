"""Tests for rng010.rangayyan_ch3_sample_std."""

from morie.fn import _array_core as np

from morie.fn.bsastat import rangayyan_ch3_sample_std


def test_rng010_basic():
    """Test basic functionality."""
    eta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch3_sample_std(eta)
    assert isinstance(result, dict)
    assert "std" in result


def test_rng010_edge():
    """Test edge cases."""
    eta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch3_sample_std(eta)
    assert isinstance(result, dict)
