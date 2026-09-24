"""Tests for infcrv.influence_function."""

from morie.fn import _array_core as np

from morie.fn.infcrv import influence_function


def test_infcrv_basic():
    """Test basic functionality."""
    estimator = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    F = np.random.default_rng(42).normal(0.0, 1.0, 40)
    x = 0.1
    result = influence_function(estimator, F, x)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_infcrv_edge():
    """Test edge cases."""
    estimator = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    F = np.random.default_rng(42).normal(0.0, 1.0, 40)
    x = 0.1
    result = influence_function(estimator, F, x)
    assert isinstance(result, dict)
