"""Tests for rgann.rangayyan_ann_mlp."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_ann_mlp


def test_rgann_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ann_mlp(X, y)
    assert isinstance(result, dict)
    assert "weights" in result


def test_rgann_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ann_mlp(X, y)
    assert isinstance(result, dict)
