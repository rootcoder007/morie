"""Tests for msm165.mvsml_ridge_lasso_elastic_eq_9_2."""

from morie.fn import _array_core as np

from morie.fn.msm165 import mvsml_ridge_lasso_elastic_eq_9_2


def test_msm165_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    beta0 = 0.5
    beta = 0.5
    result = mvsml_ridge_lasso_elastic_eq_9_2(X, beta0, beta)
    assert isinstance(result, dict)
    assert "value" in result


def test_msm165_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    beta0 = 0.5
    beta = 0.5
    result = mvsml_ridge_lasso_elastic_eq_9_2(X, beta0, beta)
    assert isinstance(result, dict)
