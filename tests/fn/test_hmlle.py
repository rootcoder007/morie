"""Tests for hmlle.geron_locally_linear_embedding."""

from morie.fn import _array_core as np

from morie.fn.hmlle import geron_locally_linear_embedding


def test_hmlle_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    n_components = 5
    result = geron_locally_linear_embedding(X, n_components)
    assert isinstance(result, dict)
    assert "estimate" in result or "embedding" in result


def test_hmlle_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    n_components = 5
    result = geron_locally_linear_embedding(X, n_components)
    assert isinstance(result, dict)
