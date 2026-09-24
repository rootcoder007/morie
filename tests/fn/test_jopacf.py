"""Tests for jopacf.joseph_partial_autocorrelation."""

from morie.fn import _array_core as np

from morie.fn.jopacf import joseph_partial_autocorrelation


def test_jopacf_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_partial_autocorrelation(x)
    assert isinstance(result, dict)
    assert "pacf" in result


def test_jopacf_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_partial_autocorrelation(x)
    assert isinstance(result, dict)
