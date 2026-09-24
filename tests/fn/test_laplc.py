"""Tests for laplc.laplace_mechanism."""

from morie.fn import _array_core as np

from morie.fn.laplc import laplace_mechanism


def test_laplc_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = laplace_mechanism(y)
    assert isinstance(result, dict)
    assert "release" in result


def test_laplc_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = laplace_mechanism(y)
    assert isinstance(result, dict)
