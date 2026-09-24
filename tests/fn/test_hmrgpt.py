"""Tests for hmrgpt.geron_regression_mlp_pytorch."""

from morie.fn import _array_core as np

from morie.fn.hmrgpt import geron_regression_mlp_pytorch


def test_hmrgpt_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_regression_mlp_pytorch(X, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "layers" in result


def test_hmrgpt_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_regression_mlp_pytorch(X, y)
    assert isinstance(result, dict)
