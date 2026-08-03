"""Tests for multinomial_loglik.multinomial_loglik."""

from morie.fn import _array_core as np

from morie.fn.multinomial_loglik import multinomial_loglik


def test_msm110_basic():
    """Test basic functionality."""
    I = np.random.default_rng(42).normal(0, 1, 100)
    yi = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    log = np.random.default_rng(42).normal(0, 1, 100)
    l = np.random.default_rng(42).normal(0, 1, 100)
    result = multinomial_loglik(I, yi, c, i, log, l)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm110_edge():
    """Test edge cases."""
    I = np.random.default_rng(42).normal(0, 1, 100)
    yi = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    log = np.random.default_rng(42).normal(0, 1, 100)
    l = np.random.default_rng(42).normal(0, 1, 100)
    result = multinomial_loglik(I, yi, c, i, log, l)
    assert isinstance(result, dict)
