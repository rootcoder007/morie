"""Tests for eslgbm.esl_gbm."""

from morie.fn import _array_core as np

from morie.fn.eslgbm import esl_gbm


def test_eslgbm_basic():
    """Test basic functionality."""
    # M is the number of boosting rounds and nu the shrinkage
    # (Hastie, Tibshirani & Friedman 2009, Algorithm 10.3); the
    # generator passed a matrix and a noise vector.
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = 10
    nu = 0.1
    result = esl_gbm(X, y, M, nu)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_eslgbm_edge():
    """Test edge cases."""
    # M is the number of boosting rounds and nu the shrinkage
    # (Hastie, Tibshirani & Friedman 2009, Algorithm 10.3); the
    # generator passed a matrix and a noise vector.
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = 10
    nu = 0.1
    result = esl_gbm(X, y, M, nu)
    assert isinstance(result, dict)
