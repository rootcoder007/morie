"""Tests for hmovr.geron_one_vs_rest."""

from morie.fn import _array_core as np

from morie.fn.hmovr import geron_one_vs_rest


def test_hmovr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_one_vs_rest(X, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "predict" in result


def test_hmovr_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_one_vs_rest(X, y)
    assert isinstance(result, dict)
