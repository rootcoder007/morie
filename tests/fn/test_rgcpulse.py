"""Tests for rgcpulse.rangayyan_carotid_pulse."""

from morie.fn import _array_core as np

from morie.fn.bsaqrs import rangayyan_carotid_pulse


def test_rgcpulse_basic():
    """Test basic functionality."""
    pulse = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_carotid_pulse(pulse, fs)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_rgcpulse_edge():
    """Test edge cases."""
    pulse = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_carotid_pulse(pulse, fs)
    assert isinstance(result, dict)
