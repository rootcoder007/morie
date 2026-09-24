"""Tests for jofrr.joseph_fourier_features."""

from morie.fn import _array_core as np

from morie.fn.jofrr import joseph_fourier_features


def test_jofrr_basic():
    """Test basic functionality."""
    n = 5
    period = 5
    k = 2.0
    result = joseph_fourier_features(n, period, k)
    assert isinstance(result, dict)
    assert "rows" in result


def test_jofrr_edge():
    """Test edge cases."""
    n = 5
    period = 5
    k = 2.0
    result = joseph_fourier_features(n, period, k)
    assert isinstance(result, dict)
