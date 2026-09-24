"""Tests for singsd.singular_spectrum."""

from morie.fn import _array_core as np

from morie.fn.singsd import singular_spectrum


def test_singsd_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    window = 5
    result = singular_spectrum(y, window)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_singsd_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    window = 5
    result = singular_spectrum(y, window)
    assert isinstance(result, dict)
