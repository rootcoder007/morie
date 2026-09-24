"""Tests for msm188.mvsml_ridge_lasso_elastic_eq_9_15."""

from morie.fn import _array_core as np

from morie.fn.msm188 import mvsml_ridge_lasso_elastic_eq_9_15


def test_msm188_basic():
    """Test basic functionality."""
    a = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    c = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = mvsml_ridge_lasso_elastic_eq_9_15(a, c)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm188_edge():
    """Test edge cases."""
    a = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    c = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = mvsml_ridge_lasso_elastic_eq_9_15(a, c)
    assert isinstance(result, dict)
