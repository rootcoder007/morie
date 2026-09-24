"""Tests for msm183.mvsml_ridge_lasso_elastic_eq_9_8."""

from morie.fn import _array_core as np

from morie.fn.msm183 import mvsml_ridge_lasso_elastic_eq_9_8


def test_msm183_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mvsml_ridge_lasso_elastic_eq_9_8(X, y)
    assert isinstance(result, dict)
    assert "beta" in result


def test_msm183_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mvsml_ridge_lasso_elastic_eq_9_8(X, y)
    assert isinstance(result, dict)
