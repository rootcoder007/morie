"""Tests for msm211.mvsml_ridge_lasso_elastic_eq_9_31."""

from morie.fn import _array_core as np

from morie.fn.msm211 import mvsml_ridge_lasso_elastic_eq_9_31


def test_msm211_basic():
    """Test basic functionality."""
    alpha = 0.1
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mvsml_ridge_lasso_elastic_eq_9_31(alpha, X, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm211_edge():
    """Test edge cases."""
    alpha = 0.1
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mvsml_ridge_lasso_elastic_eq_9_31(alpha, X, y)
    assert isinstance(result, dict)
