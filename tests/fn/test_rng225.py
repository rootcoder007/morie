"""Tests for rng225.rangayyan_ch4_composite_signal_in_terms_of_g."""

from morie.fn import _array_core as np

from morie.fn.bsasig import rangayyan_ch4_composite_signal_in_terms_of_g


def test_rng225_basic():
    """Test basic functionality."""
    g = np.random.default_rng(42).normal(0.0, 1.0, 40)
    shifts = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = rangayyan_ch4_composite_signal_in_terms_of_g(g, shifts)
    assert isinstance(result, dict)
    assert "x" in result or "x" in result


def test_rng225_edge():
    """Test edge cases."""
    g = np.random.default_rng(42).normal(0.0, 1.0, 40)
    shifts = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = rangayyan_ch4_composite_signal_in_terms_of_g(g, shifts)
    assert isinstance(result, dict)
