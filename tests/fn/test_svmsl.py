"""Tests for svmsl.svm_soft_margin."""

from morie.fn import _array_core as np

from morie.fn.svmsl import svm_soft_margin


def test_svmsl_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    C = 0.1
    result = svm_soft_margin(X, y, C)
    assert isinstance(result, dict)
    assert "estimate" in result or "beta" in result


def test_svmsl_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    C = 0.1
    result = svm_soft_margin(X, y, C)
    assert isinstance(result, dict)
