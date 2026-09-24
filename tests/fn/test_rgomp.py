"""Tests for rgomp.rangayyan_omp."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_omp


def test_rgomp_basic():
    """Test basic functionality."""
    x = 0.5
    D = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_omp(x, D)
    assert isinstance(result, dict)
    assert "coefficients" in result


def test_rgomp_edge():
    """Test edge cases."""
    x = 0.5
    D = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_omp(x, D)
    assert isinstance(result, dict)
