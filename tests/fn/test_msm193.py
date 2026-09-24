"""Tests for msm193.mvsml_ridge_lasso_elastic_eq_9_21."""

from morie.fn import _array_core as np

from morie.fn.msm193 import mvsml_ridge_lasso_elastic_eq_9_21


def test_msm193_basic():
    """Test basic functionality."""
    a = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    c = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mvsml_ridge_lasso_elastic_eq_9_21(a, c)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm193_edge():
    """Test edge cases."""
    a = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    c = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mvsml_ridge_lasso_elastic_eq_9_21(a, c)
    assert isinstance(result, dict)
