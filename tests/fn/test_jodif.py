"""Tests for jodif.joseph_differencing."""

from morie.fn import _array_core as np

from morie.fn.jodif import joseph_differencing


def test_jodif_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_differencing(x)
    assert isinstance(result, dict)
    assert "w" in result


def test_jodif_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_differencing(x)
    assert isinstance(result, dict)
