"""Tests for rgcpulse.rangayyan_carotid_pulse."""

from morie.fn import _array_core as np

from morie.fn.bsaqrs import rangayyan_carotid_pulse


def test_rgcpulse_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    pulse = rng.normal(0, 1, 100)
    fs = 100.0
    qrs = [10, 40, 70]
    result = rangayyan_carotid_pulse(pulse, fs, qrs)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_rgcpulse_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    pulse = rng.normal(0, 1, 100)
    fs = 100.0
    qrs = [30, 70]
    result = rangayyan_carotid_pulse(pulse, fs, qrs)
    assert isinstance(result, dict)
    assert len(result) > 0
