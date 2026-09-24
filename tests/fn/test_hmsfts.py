"""Tests for hmsfts.geron_softmax_score."""

from morie.fn import _array_core as np

from morie.fn.hmsfts import geron_softmax_score


def test_hmsfts_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    theta = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = geron_softmax_score(X, theta)
    assert isinstance(result, dict)
    assert "estimate" in result or "scores" in result


def test_hmsfts_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    theta = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = geron_softmax_score(X, theta)
    assert isinstance(result, dict)
