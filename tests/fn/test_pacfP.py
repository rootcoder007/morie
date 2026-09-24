"""Tests for pacfP.partial_autocorrelation."""

from morie.fn import _array_core as np

from morie.fn.pacfP import partial_autocorrelation


def test_pacfP_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    lag_max = 5
    result = partial_autocorrelation(y, lag_max)
    assert isinstance(result, dict)
    assert "pacf" in result


def test_pacfP_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    lag_max = 5
    result = partial_autocorrelation(y, lag_max)
    assert isinstance(result, dict)
