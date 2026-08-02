"""Tests for eslvbt.esl_var_beta_hat."""

from morie.fn import _array_core as np

from morie.fn.eslvbt import esl_var_beta_hat


def test_eslvbt_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    sigma2 = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_var_beta_hat(X, sigma2)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_eslvbt_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    sigma2 = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_var_beta_hat(X, sigma2)
    assert isinstance(result, dict)
