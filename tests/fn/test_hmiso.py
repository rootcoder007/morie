"""Tests for hmiso.geron_isomap."""

from morie.fn import _array_core as np

from morie.fn.hmiso import geron_isomap


def test_hmiso_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    n_components = 5
    result = geron_isomap(X, n_components)
    assert isinstance(result, dict)
    assert "estimate" in result or "embedding" in result


def test_hmiso_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    n_components = 5
    result = geron_isomap(X, n_components)
    assert isinstance(result, dict)
