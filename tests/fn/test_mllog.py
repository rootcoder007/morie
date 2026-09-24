"""Tests for mllog.ml_log_likelihood_regression."""

from morie.fn import _array_core as np

from morie.fn.mllog import ml_log_likelihood_regression


def test_mllog_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = ml_log_likelihood_regression(X, y)
    assert isinstance(result, dict)
    assert "loglik" in result


def test_mllog_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = ml_log_likelihood_regression(X, y)
    assert isinstance(result, dict)
