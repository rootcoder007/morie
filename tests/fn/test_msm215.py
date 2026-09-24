"""Tests for msm215.mvsml_ridge_lasso_elastic_eq_9_33."""

from morie.fn import _array_core as np

from morie.fn.msm215 import mvsml_ridge_lasso_elastic_eq_9_33


def test_msm215_basic():
    """Test basic functionality."""
    alpha = 0.1
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mvsml_ridge_lasso_elastic_eq_9_33(alpha, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm215_edge():
    """Test edge cases."""
    alpha = 0.1
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mvsml_ridge_lasso_elastic_eq_9_33(alpha, y)
    assert isinstance(result, dict)
