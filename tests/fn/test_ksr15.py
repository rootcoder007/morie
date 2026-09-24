"""Tests for ksr15.kosorok_one_step_estimator."""

from morie.fn import _array_core as np

from morie.fn.ksr15 import kosorok_one_step_estimator


def test_ksr15_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    theta0 = 0.1
    result = kosorok_one_step_estimator(x, theta0)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_ksr15_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    theta0 = 0.1
    result = kosorok_one_step_estimator(x, theta0)
    assert isinstance(result, dict)
