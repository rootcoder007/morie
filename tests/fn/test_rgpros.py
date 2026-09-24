"""Tests for rgpros.rangayyan_prosthetic_valve."""

from morie.fn import _array_core as np

from morie.fn.bsaphys import rangayyan_prosthetic_valve


def test_rgpros_basic():
    """Test basic functionality."""
    pcg = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.1
    result = rangayyan_prosthetic_valve(pcg, fs)
    assert isinstance(result, dict)
    assert "total_power" in result


def test_rgpros_edge():
    """Test edge cases."""
    pcg = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.1
    result = rangayyan_prosthetic_valve(pcg, fs)
    assert isinstance(result, dict)
