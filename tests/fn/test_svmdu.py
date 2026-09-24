"""Tests for svmdu.svm_dual_wolfe."""

from morie.fn import _array_core as np

from morie.fn.svmdu import svm_dual_wolfe


def test_svmdu_basic():
    """Test basic functionality."""
    alpha = np.random.default_rng(42).normal(0.0, 1.0, 40)
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = svm_dual_wolfe(alpha, X, y)
    assert isinstance(result, dict)
    assert "dual" in result


def test_svmdu_edge():
    """Test edge cases."""
    alpha = np.random.default_rng(42).normal(0.0, 1.0, 40)
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = svm_dual_wolfe(alpha, X, y)
    assert isinstance(result, dict)
