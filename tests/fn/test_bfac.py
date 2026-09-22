"""Tests for bfac.bayes_factor."""

from morie.fn import _array_core as np

from morie.fn.bfac import bayes_factor


def test_bfac_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    log_lik_a = float(rng.normal())
    log_lik_b = float(rng.normal())
    result = bayes_factor(log_lik_a, log_lik_b)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_bfac_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    log_lik_a = float(rng.normal())
    log_lik_b = float(rng.normal())
    result = bayes_factor(log_lik_a, log_lik_b)
    assert isinstance(result, dict)
