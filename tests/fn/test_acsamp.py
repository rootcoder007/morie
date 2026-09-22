"""Tests for acsamp.sample_autocorrelation."""

from morie.fn import _array_core as np

from morie.fn.acsamp import sample_autocorrelation


def test_acsamp_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = sample_autocorrelation(y)
    assert isinstance(result, dict)
    assert "acf" in result
def test_acsamp_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = sample_autocorrelation(y)
    assert isinstance(result, dict)
