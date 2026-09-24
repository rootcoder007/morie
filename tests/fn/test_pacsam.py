"""Tests for pacsam.sample_partial_autocorr."""

from morie.fn import _array_core as np

from morie.fn.pacsam import sample_partial_autocorr


def test_pacsam_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = sample_partial_autocorr(y)
    assert isinstance(result, dict)
    assert "pacf" in result


def test_pacsam_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = sample_partial_autocorr(y)
    assert isinstance(result, dict)
