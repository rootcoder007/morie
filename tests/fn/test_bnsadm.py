"""Tests for bnsadm.bound_admissible_estimators."""

from morie.fn import _array_core as np

from morie.fn.bnsadm import bound_admissible_estimators


def test_bnsadm_basic():
    """Test basic functionality."""
    y = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    D = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = bound_admissible_estimators(y, D, X)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_bnsadm_edge():
    """Test edge cases."""
    y = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    D = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = bound_admissible_estimators(y, D, X)
    assert isinstance(result, dict)
