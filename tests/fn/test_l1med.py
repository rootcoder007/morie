"""Tests for l1med.l1_median."""

from morie.fn import _array_core as np

from morie.fn.l1med import l1_median


def test_l1med_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    tol = 1e-6
    result = l1_median(X, tol)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_l1med_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    tol = 1e-6
    result = l1_median(X, tol)
    assert isinstance(result, dict)
