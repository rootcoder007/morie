"""Tests for hmins.geron_instance_based."""

from morie.fn import _array_core as np

from morie.fn.hmins import geron_instance_based


def test_hmins_basic():
    """Test basic functionality."""
    X_train = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y_train = np.random.default_rng(42).normal(0.0, 1.0, 40)
    x_query = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = geron_instance_based(X_train, y_train, x_query)
    assert isinstance(result, dict)
    assert "estimate" in result or "prediction" in result


def test_hmins_edge():
    """Test edge cases."""
    X_train = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y_train = np.random.default_rng(42).normal(0.0, 1.0, 40)
    x_query = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = geron_instance_based(X_train, y_train, x_query)
    assert isinstance(result, dict)
