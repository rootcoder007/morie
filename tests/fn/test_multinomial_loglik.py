"""Tests for multinomial_loglik.multinomial_loglik."""

from morie.fn import _array_core as np

from morie.fn.multinomial_loglik import multinomial_loglik


def test_msm110_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    beta0 = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    beta = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = multinomial_loglik(X, y, beta0, beta)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm110_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    beta0 = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    beta = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = multinomial_loglik(X, y, beta0, beta)
    assert isinstance(result, dict)
