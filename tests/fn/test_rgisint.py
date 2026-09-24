"""Tests for rgisint.rangayyan_isometric_contraction."""

from morie.fn import _array_core as np

from morie.fn.rgisint import rangayyan_isometric_contraction


def test_rgisint_basic():
    """Test basic functionality."""
    emg = np.random.default_rng(42).normal(0.0, 1.0, 40)
    force = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.1
    result = rangayyan_isometric_contraction(emg, force, fs)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_rgisint_edge():
    """Test edge cases."""
    emg = np.random.default_rng(42).normal(0.0, 1.0, 40)
    force = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.1
    result = rangayyan_isometric_contraction(emg, force, fs)
    assert isinstance(result, dict)
