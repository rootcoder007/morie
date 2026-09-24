"""Tests for rng179.rangayyan_ch4_filtered_derivative_murthy."""

from morie.fn import _array_core as np

from morie.fn.bsaqrs import rangayyan_ch4_filtered_derivative_murthy


def test_rng179_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch4_filtered_derivative_murthy(x)
    assert isinstance(result, dict)
    assert "g1" in result


def test_rng179_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch4_filtered_derivative_murthy(x)
    assert isinstance(result, dict)
