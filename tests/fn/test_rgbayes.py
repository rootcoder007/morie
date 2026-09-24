"""Tests for rgbayes.rangayyan_bayes_classifier."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_bayes_classifier


def test_rgbayes_basic():
    """Test basic functionality."""
    likelihoods = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = rangayyan_bayes_classifier(likelihoods)
    assert isinstance(result, dict)
    assert "d" in result


def test_rgbayes_edge():
    """Test edge cases."""
    likelihoods = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = rangayyan_bayes_classifier(likelihoods)
    assert isinstance(result, dict)
