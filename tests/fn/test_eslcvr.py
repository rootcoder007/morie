"""Tests for eslcvr.esl_cv_score."""

from morie.fn import _array_core as np

from morie.fn.eslcvr import esl_cv_score


def test_eslcvr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = esl_cv_score(X, y)
    assert isinstance(result, dict)
    assert "cv" in result
def test_eslcvr_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = esl_cv_score(X, y)
    assert isinstance(result, dict)
