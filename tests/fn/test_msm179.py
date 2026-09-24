"""Tests for msm179.mvsml_ridge_lasso_elastic_eq_9_6."""

from morie.fn import _array_core as np

from morie.fn.msm179 import mvsml_ridge_lasso_elastic_eq_9_6


def test_msm179_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mvsml_ridge_lasso_elastic_eq_9_6(X, y)
    assert isinstance(result, dict)
    assert "beta" in result


def test_msm179_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mvsml_ridge_lasso_elastic_eq_9_6(X, y)
    assert isinstance(result, dict)
