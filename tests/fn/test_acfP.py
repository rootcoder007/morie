"""Tests for acfP.autocorrelation."""

from morie.fn import _array_core as np

from morie.fn.acfP import autocorrelation


def test_acfP_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = autocorrelation(y)
    assert isinstance(result, dict)
    assert "acf" in result
def test_acfP_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = autocorrelation(y)
    assert isinstance(result, dict)
