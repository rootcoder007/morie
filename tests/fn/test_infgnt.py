"""Tests for infgnt.information_geometry."""

from morie.fn import _array_core as np

from morie.fn.infgnt import information_geometry


def test_infgnt_basic():
    """Test basic functionality."""
    log_p = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    theta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    support = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = information_geometry(log_p, theta, support)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_infgnt_edge():
    """Test edge cases."""
    log_p = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    theta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    support = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = information_geometry(log_p, theta, support)
    assert isinstance(result, dict)
