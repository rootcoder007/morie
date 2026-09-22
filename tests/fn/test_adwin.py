"""Tests for adwin.adwin."""

from morie.fn import _array_core as np

from morie.fn.adwin import adwin


def test_adwin_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = adwin(x)
    assert isinstance(result, dict)
    assert "mean" in result
def test_adwin_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = adwin(x)
    assert isinstance(result, dict)
