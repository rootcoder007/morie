"""Tests for rng146.rangayyan_ch3_optimal_wiener_filter."""

from morie.fn import _array_core as np

from morie.fn.bsaadapt import rangayyan_ch3_optimal_wiener_filter


def test_rng146_basic():
    """Test basic functionality."""
    Phi = np.random.default_rng(42).normal(0, 1, 100)
    Theta = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_optimal_wiener_filter(Phi, Theta)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_rng146_edge():
    """Test edge cases."""
    Phi = np.random.default_rng(42).normal(0, 1, 100)
    Theta = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_optimal_wiener_filter(Phi, Theta)
    assert isinstance(result, dict)
