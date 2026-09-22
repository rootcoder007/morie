"""Tests for eslsig.esl_residual_variance."""

from morie.fn import _array_core as np

from morie.fn.eslsig import esl_residual_variance


def test_eslsig_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = esl_residual_variance(X, y)
    assert isinstance(result, dict)
    assert "value" in result
def test_eslsig_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = esl_residual_variance(X, y)
    assert isinstance(result, dict)
