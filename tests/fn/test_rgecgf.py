"""Tests for rgecgf.rangayyan_ecg_features."""

from morie.fn import _array_core as np

from morie.fn.bsaqrs import rangayyan_ecg_features


def test_rgecgf_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    qrs = 5
    fs = 0.1
    result = rangayyan_ecg_features(x, qrs, fs)
    assert isinstance(result, dict)
    assert "pamp" in result


def test_rgecgf_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    qrs = 5
    fs = 0.1
    result = rangayyan_ecg_features(x, qrs, fs)
    assert isinstance(result, dict)
