"""Tests for rng094.rangayyan_ch3_hann_frequency_response_simplified."""

from morie.fn import _array_core as np

from morie.fn.bsafilt import rangayyan_ch3_hann_frequency_response_simplified


def test_rng094_basic():
    """Test basic functionality."""
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_hann_frequency_response_simplified(omega)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_rng094_edge():
    """Test edge cases."""
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_hann_frequency_response_simplified(omega)
    assert isinstance(result, dict)
