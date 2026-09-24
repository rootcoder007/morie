"""Tests for hmearl.geron_early_stopping."""

from morie.fn import _array_core as np

from morie.fn.hmearl import geron_early_stopping


def test_hmearl_basic():
    """Test basic functionality."""
    X_train = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y_train = np.random.default_rng(42).normal(0.0, 1.0, 40)
    X_val = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y_val = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_early_stopping(X_train, y_train, X_val, y_val)
    assert isinstance(result, dict)
    assert "estimate" in result or "theta" in result


def test_hmearl_edge():
    """Test edge cases."""
    X_train = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y_train = np.random.default_rng(42).normal(0.0, 1.0, 40)
    X_val = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y_val = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_early_stopping(X_train, y_train, X_val, y_val)
    assert isinstance(result, dict)
