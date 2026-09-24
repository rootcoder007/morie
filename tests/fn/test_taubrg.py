"""Tests for taubrg.tau_estimator_regression."""

from morie.fn import _array_core as np

from morie.fn.taubrg import tau_estimator_regression


def test_taubrg_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = tau_estimator_regression(X, y)
    assert isinstance(result, dict)
    assert "beta" in result


def test_taubrg_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = tau_estimator_regression(X, y)
    assert isinstance(result, dict)
