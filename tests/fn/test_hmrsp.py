"""Tests for hmrsp.geron_random_subspaces."""

from morie.fn import _array_core as np

from morie.fn.hmrsp import geron_random_subspaces


def test_hmrsp_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_random_subspaces(X, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "predict" in result


def test_hmrsp_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_random_subspaces(X, y)
    assert isinstance(result, dict)
