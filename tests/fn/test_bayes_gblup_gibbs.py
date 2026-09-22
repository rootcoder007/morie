"""Tests for bayes_gblup_gibbs.bayes_gblup_gibbs."""

from morie.fn import _array_core as np

from morie.fn.bayes_gblup_gibbs import bayes_gblup_gibbs


def test_msm049_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0, 1, 100)
    G = np.eye(10)
    result = bayes_gblup_gibbs(y, G)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm049_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0, 1, 100)
    G = np.eye(10)
    result = bayes_gblup_gibbs(y, G)
    assert isinstance(result, dict)
