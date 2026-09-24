"""Tests for msm190.mvsml_ridge_lasso_elastic_eq_9_18."""

from morie.fn import _array_core as np

from morie.fn.msm190 import mvsml_ridge_lasso_elastic_eq_9_18


def test_msm190_basic():
    """Test basic functionality."""
    a = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    c = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_ridge_lasso_elastic_eq_9_18(a, c)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm190_edge():
    """Test edge cases."""
    a = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    c = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_ridge_lasso_elastic_eq_9_18(a, c)
    assert isinstance(result, dict)
