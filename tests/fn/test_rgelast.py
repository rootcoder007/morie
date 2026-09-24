"""Tests for rgelast.rangayyan_heart_elasticity."""

from morie.fn import _array_core as np

from morie.fn.bsaphys import rangayyan_heart_elasticity


def test_rgelast_basic():
    """Test basic functionality."""
    pcg = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.1
    result = rangayyan_heart_elasticity(pcg, fs)
    assert isinstance(result, dict)
    assert "dominant_frequency" in result


def test_rgelast_edge():
    """Test edge cases."""
    pcg = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.1
    result = rangayyan_heart_elasticity(pcg, fs)
    assert isinstance(result, dict)
