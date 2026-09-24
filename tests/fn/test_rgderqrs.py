"""Tests for rgderqrs.rangayyan_deriv_qrs."""

from morie.fn import _array_core as np

from morie.fn.bsaqrs import rangayyan_deriv_qrs


def test_rgderqrs_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.1
    result = rangayyan_deriv_qrs(x, fs)
    assert isinstance(result, dict)
    assert "qrs" in result


def test_rgderqrs_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.1
    result = rangayyan_deriv_qrs(x, fs)
    assert isinstance(result, dict)
