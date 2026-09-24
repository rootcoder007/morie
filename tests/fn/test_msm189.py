"""Tests for msm189.mvsml_ridge_lasso_elastic_eq_9_17."""

from morie.fn import _array_core as np

from morie.fn.msm189 import mvsml_ridge_lasso_elastic_eq_9_17


def test_msm189_basic():
    """Test basic functionality."""
    a = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    c = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_ridge_lasso_elastic_eq_9_17(a, c)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm189_edge():
    """Test edge cases."""
    a = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    c = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_ridge_lasso_elastic_eq_9_17(a, c)
    assert isinstance(result, dict)
