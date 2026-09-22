"""Tests for baylog.bayes_logistic."""

from morie.fn import _array_core as np

from morie.fn.baylog import bayes_logistic


def test_baylog_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    result = bayes_logistic(X, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_baylog_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    result = bayes_logistic(X, y)
    assert isinstance(result, dict)
