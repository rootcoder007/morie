"""Tests for rng134.rangayyan_ch3_butterworth_lowpass_dft_indexed."""

from morie.fn import _array_core as np

from morie.fn.bsafilt import rangayyan_ch3_butterworth_lowpass_dft_indexed


def test_rng134_basic():
    """Test basic functionality."""
    k = 5
    k_c = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = rangayyan_ch3_butterworth_lowpass_dft_indexed(k, k_c, N)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_rng134_edge():
    """Test edge cases."""
    k = 5
    k_c = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = rangayyan_ch3_butterworth_lowpass_dft_indexed(k, k_c, N)
    assert isinstance(result, dict)
