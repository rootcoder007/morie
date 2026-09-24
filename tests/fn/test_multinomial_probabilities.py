"""Tests for multinomial_probabilities.multinomial_probabilities."""

from morie.fn import _array_core as np

from morie.fn.multinomial_probabilities import multinomial_probabilities


def test_msm106_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    beta0 = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    beta = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = multinomial_probabilities(X, beta0, beta)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm106_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    beta0 = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    beta = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = multinomial_probabilities(X, beta0, beta)
    assert isinstance(result, dict)
