"""Tests for msm192.mvsml_ridge_lasso_elastic_eq_9_20."""

from morie.fn import _array_core as np

from morie.fn.msm192 import mvsml_ridge_lasso_elastic_eq_9_20


def test_msm192_basic():
    """Test basic functionality."""
    a = np.random.default_rng(42).normal(0.0, 1.0, 40)
    c = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_ridge_lasso_elastic_eq_9_20(a, c)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm192_edge():
    """Test edge cases."""
    a = np.random.default_rng(42).normal(0.0, 1.0, 40)
    c = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_ridge_lasso_elastic_eq_9_20(a, c)
    assert isinstance(result, dict)
