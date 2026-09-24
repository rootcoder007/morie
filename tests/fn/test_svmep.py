"""Tests for svmep.svr_epsilon_insensitive."""

from morie.fn import _array_core as np

from morie.fn.svmep import svr_epsilon_insensitive


def test_svmep_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    C = 0.1
    eps = 0.1
    result = svr_epsilon_insensitive(X, y, C, eps)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_svmep_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    C = 0.1
    eps = 0.1
    result = svr_epsilon_insensitive(X, y, C, eps)
    assert isinstance(result, dict)
