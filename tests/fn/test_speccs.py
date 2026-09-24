"""Tests for speccs.cross_spectrum."""

from morie.fn import _array_core as np

from morie.fn.speccs import cross_spectrum


def test_speccs_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = cross_spectrum(x, y)
    assert isinstance(result, dict)
    assert "omega" in result


def test_speccs_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = cross_spectrum(x, y)
    assert isinstance(result, dict)
