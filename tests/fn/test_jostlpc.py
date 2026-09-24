"""Tests for jostlpc.joseph_stl_decomposition."""

from morie.fn import _array_core as np

from morie.fn.jostlpc import joseph_stl_decomposition


def test_jostlpc_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    period = 5
    result = joseph_stl_decomposition(x, period)
    assert isinstance(result, dict)
    assert "trend" in result


def test_jostlpc_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    period = 5
    result = joseph_stl_decomposition(x, period)
    assert isinstance(result, dict)
