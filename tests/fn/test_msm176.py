"""Tests for msm176.mvsml_ridge_lasso_elastic_eq_9_4."""

from morie.fn import _array_core as np

from morie.fn.msm176 import mvsml_ridge_lasso_elastic_eq_9_4


def test_msm176_basic():
    """Test basic functionality."""
    X = [[1.0, 1.0], [-1.0, -1.0]]
    beta0 = 1.0
    beta = [2.0, 3.0]
    result = mvsml_ridge_lasso_elastic_eq_9_4(X, beta0, beta)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm176_edge():
    """Test edge cases."""
    X = [[1.0, 1.0], [-1.0, -1.0]]
    beta0 = 1.0
    beta = [2.0, 3.0]
    result = mvsml_ridge_lasso_elastic_eq_9_4(X, beta0, beta)
    assert isinstance(result, dict)
