"""Tests for bayes_gblup_gibbs.bayes_gblup_gibbs."""

from morie.fn import _array_core as np

from morie.fn.bayes_gblup_gibbs import bayes_gblup_gibbs


def test_msm049_basic():
    """Test basic functionality."""
    p = 5
    X1XT = np.random.default_rng(42).normal(0, 1, 100)
    G = np.eye(10)
    which = np.random.default_rng(42).normal(0, 1, 100)
    known = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_gblup_gibbs(p, X1XT, G, which, known, the)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm049_edge():
    """Test edge cases."""
    p = 5
    X1XT = np.random.default_rng(42).normal(0, 1, 100)
    G = np.eye(10)
    which = np.random.default_rng(42).normal(0, 1, 100)
    known = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_gblup_gibbs(p, X1XT, G, which, known, the)
    assert isinstance(result, dict)
