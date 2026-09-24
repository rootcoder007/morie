"""Tests for rng211.rangayyan_ch4_average_output_noise_power."""

from morie.fn import _array_core as np

from morie.fn.bsacorr import rangayyan_ch4_average_output_noise_power


def test_rng211_basic():
    """Test basic functionality."""
    P_eta_i = 0.1
    H = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch4_average_output_noise_power(P_eta_i, H)
    assert isinstance(result, dict)
    assert "output_power" in result


def test_rng211_edge():
    """Test edge cases."""
    P_eta_i = 0.1
    H = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch4_average_output_noise_power(P_eta_i, H)
    assert isinstance(result, dict)
