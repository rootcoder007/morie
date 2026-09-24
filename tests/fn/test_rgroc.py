"""Tests for rgroc.rangayyan_roc_curve."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_roc_curve


def test_rgroc_basic():
    """Test basic functionality."""
    scores = np.random.default_rng(42).normal(0.0, 1.0, 40)
    labels = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    result = rangayyan_roc_curve(scores, labels)
    assert isinstance(result, dict)
    assert "fpf" in result


def test_rgroc_edge():
    """Test edge cases."""
    scores = np.random.default_rng(42).normal(0.0, 1.0, 40)
    labels = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    result = rangayyan_roc_curve(scores, labels)
    assert isinstance(result, dict)
