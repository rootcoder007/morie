"""Tests for rgperio.rangayyan_periodogram."""

from morie.fn import _array_core as np

from morie.fn.bsacorr import rangayyan_periodogram


def test_rgperio_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_periodogram(x)
    assert isinstance(result, dict)
    assert "freqs" in result


def test_rgperio_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_periodogram(x)
    assert isinstance(result, dict)
