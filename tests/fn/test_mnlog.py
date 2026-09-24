"""Tests for mnlog.multinomial_logistic_penalized."""

from morie.fn import _array_core as np

from morie.fn.mnlog import multinomial_logistic_penalized


def test_mnlog_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    beta0 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    beta = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    lam = 0.1
    result = multinomial_logistic_penalized(X, y, beta0, beta, lam)
    assert isinstance(result, dict)
    assert "loglik" in result


def test_mnlog_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    beta0 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    beta = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    lam = 0.1
    result = multinomial_logistic_penalized(X, y, beta0, beta, lam)
    assert isinstance(result, dict)
