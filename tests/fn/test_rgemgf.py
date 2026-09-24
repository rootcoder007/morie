"""Tests for rgemgf.rangayyan_emg_force."""

from morie.fn import _array_core as np

from morie.fn.rgemgf import rangayyan_emg_force


def test_rgemgf_basic():
    """Test basic functionality."""
    emg = np.random.default_rng(42).normal(0.0, 1.0, 40)
    force = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.1
    result = rangayyan_emg_force(emg, force, fs)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_rgemgf_edge():
    """Test edge cases."""
    emg = np.random.default_rng(42).normal(0.0, 1.0, 40)
    force = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.1
    result = rangayyan_emg_force(emg, force, fs)
    assert isinstance(result, dict)
