"""Tests for multinomial_probabilities.multinomial_probabilities."""

from morie.fn import _array_core as np

from morie.fn.multinomial_probabilities import multinomial_probabilities


def test_msm106_basic():
    """Test basic functionality."""
    C = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    following = np.random.default_rng(42).normal(0, 1, 100)
    exp = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    result = multinomial_probabilities(C, the, following, exp, xT, i)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm106_edge():
    """Test edge cases."""
    C = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    following = np.random.default_rng(42).normal(0, 1, 100)
    exp = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    result = multinomial_probabilities(C, the, following, exp, xT, i)
    assert isinstance(result, dict)
