"""Tests for rgwelch.rangayyan_welch_psd."""

from morie.fn import _array_core as np

from morie.fn.bsacorr import rangayyan_welch_psd


def test_rgwelch_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_welch_psd(x)
    assert isinstance(result, dict)
    assert "freqs" in result


def test_rgwelch_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_welch_psd(x)
    assert isinstance(result, dict)
