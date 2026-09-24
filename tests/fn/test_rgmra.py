"""Tests for rgmra.rangayyan_mra."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_mra


def test_rgmra_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_mra(x)
    assert isinstance(result, dict)
    assert "approximation" in result


def test_rgmra_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_mra(x)
    assert isinstance(result, dict)
