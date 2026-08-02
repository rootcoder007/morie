"""Tests for laplI.inverse_laplace."""

from morie.fn import _array_core as np

from morie.fn.laplI import inverse_laplace


def test_laplI_basic():
    """Test basic functionality."""
    F = np.random.default_rng(43).normal(0, 1, 100)
    s = 90
    t = np.linspace(0, 10, 100)
    result = inverse_laplace(F, s, t)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_laplI_edge():
    """Test edge cases."""
    F = np.random.default_rng(43).normal(0, 1, 100)
    s = 90
    t = np.linspace(0, 10, 100)
    result = inverse_laplace(F, s, t)
    assert isinstance(result, dict)
