"""Tests for hmonl.geron_online_learning."""

from morie.fn import _array_core as np

from morie.fn.hmonl import geron_online_learning


def test_hmonl_basic():
    """Test basic functionality."""
    X_stream = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y_stream = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_online_learning(X_stream, y_stream)
    assert isinstance(result, dict)
    assert "estimate" in result or "theta" in result


def test_hmonl_edge():
    """Test edge cases."""
    X_stream = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y_stream = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_online_learning(X_stream, y_stream)
    assert isinstance(result, dict)
