"""Tests for hmovo.geron_one_vs_one."""

from morie.fn import _array_core as np

from morie.fn.hmovo import geron_one_vs_one


def test_hmovo_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_one_vs_one(X, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "predict" in result


def test_hmovo_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_one_vs_one(X, y)
    assert isinstance(result, dict)
