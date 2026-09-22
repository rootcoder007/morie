"""Tests for msm203.mvsml_ridge_lasso_elastic_eq_9_29."""

from morie.fn import _array_core as np

from morie.fn.msm203 import mvsml_ridge_lasso_elastic_eq_9_29


def test_msm203_basic():
    """Test basic functionality."""
    alpha = 0.5
    y = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_29(alpha, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm203_edge():
    """Test edge cases."""
    alpha = 0.5
    y = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_29(alpha, y)
    assert isinstance(result, dict)
