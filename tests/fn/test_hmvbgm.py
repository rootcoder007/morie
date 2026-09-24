"""Tests for hmvbgm.geron_variational_bayes_gmm."""

from morie.fn import _array_core as np

from morie.fn.hmvbgm import geron_variational_bayes_gmm


def test_hmvbgm_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_variational_bayes_gmm(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "weights" in result


def test_hmvbgm_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_variational_bayes_gmm(X)
    assert isinstance(result, dict)
