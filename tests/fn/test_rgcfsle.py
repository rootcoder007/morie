"""Tests for rgcfsle.rangayyan_coupled_freq_select."""

from morie.fn import _array_core as np

from morie.fn.bsacorr import rangayyan_coupled_freq_select


def test_rgcfsle_basic():
    """Test basic functionality."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    resp = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_coupled_freq_select(ecg, resp, fs)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_rgcfsle_edge():
    """Test edge cases."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    resp = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_coupled_freq_select(ecg, resp, fs)
    assert isinstance(result, dict)
