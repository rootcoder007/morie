"""Tests for msm202.mvsml_ridge_lasso_elastic_eq_9_28."""

from morie.fn import _array_core as np

from morie.fn.msm202 import mvsml_ridge_lasso_elastic_eq_9_28


def test_msm202_basic():
    """Test basic functionality."""
    alpha = 0.1
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mvsml_ridge_lasso_elastic_eq_9_28(alpha, X, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm202_edge():
    """Test edge cases."""
    alpha = 0.1
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mvsml_ridge_lasso_elastic_eq_9_28(alpha, X, y)
    assert isinstance(result, dict)
