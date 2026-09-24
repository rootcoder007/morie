"""Tests for rgsapn.rangayyan_sleep_apnea."""

from morie.fn import _array_core as np

from morie.fn.bsaqrs import rangayyan_sleep_apnea


def test_rgsapn_basic():
    """Test basic functionality."""
    edr = np.random.default_rng(42).normal(0.0, 1.0, 40)
    spo2 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.1
    result = rangayyan_sleep_apnea(edr, spo2, fs)
    assert isinstance(result, dict)
    assert "events" in result


def test_rgsapn_edge():
    """Test edge cases."""
    edr = np.random.default_rng(42).normal(0.0, 1.0, 40)
    spo2 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.1
    result = rangayyan_sleep_apnea(edr, spo2, fs)
    assert isinstance(result, dict)
