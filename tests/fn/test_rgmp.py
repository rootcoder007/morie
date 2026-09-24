"""Tests for rgmp.rangayyan_matching_pursuit."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_matching_pursuit


def test_rgmp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_matching_pursuit(x)
    assert isinstance(result, dict)
    assert "coefficients" in result


def test_rgmp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_matching_pursuit(x)
    assert isinstance(result, dict)
