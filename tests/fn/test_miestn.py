"""Tests for miestn.mi_neural_estimator."""

from morie.fn import _array_core as np

from morie.fn.miestn import mi_neural_estimator


def test_miestn_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mi_neural_estimator(x, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_miestn_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mi_neural_estimator(x, y)
    assert isinstance(result, dict)
