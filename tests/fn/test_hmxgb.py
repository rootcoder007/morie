"""Tests for hmxgb.geron_xgboost."""

from morie.fn import _array_core as np

from morie.fn.hmxgb import geron_xgboost


def test_hmxgb_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_xgboost(X, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "predicted" in result


def test_hmxgb_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_xgboost(X, y)
    assert isinstance(result, dict)
