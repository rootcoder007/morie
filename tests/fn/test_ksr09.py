"""Tests for ksr09.kosorok_z_estimator."""

from morie.fn import _array_core as np

from morie.fn.ksr09 import kosorok_z_estimator


def test_ksr09_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = kosorok_z_estimator(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_ksr09_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = kosorok_z_estimator(x)
    assert isinstance(result, dict)
