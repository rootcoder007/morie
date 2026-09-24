"""Tests for hmclsn.geron_classification_mlp."""

from morie.fn import _array_core as np

from morie.fn.hmclsn import geron_classification_mlp


def test_hmclsn_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_classification_mlp(X, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "weights" in result


def test_hmclsn_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_classification_mlp(X, y)
    assert isinstance(result, dict)
