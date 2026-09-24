"""Tests for aitcmd.compositional_median."""

from morie.fn import _array_core as np

from morie.fn.aitcmd import compositional_median


def test_aitcmd_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0.1, 10.0, (100, 5))
    tol = 1e-6
    result = compositional_median(X, tol)
    assert isinstance(result, dict)
    assert "estimate" in result or "median" in result or "clr_median" in result


def test_aitcmd_edge():
    """Test edge cases with small but valid strictly positive compositions."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0.5, 5.0, (20, 3))
    tol = 1e-6
    result = compositional_median(X, tol)
    assert isinstance(result, dict)
    assert "estimate" in result or "median" in result or "clr_median" in result
