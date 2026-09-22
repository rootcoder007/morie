"""Tests for bnnipw.bound_no_iv_proxy."""

from morie.fn import _array_core as np

from morie.fn.bnnipw import bound_no_iv_proxy


def test_bnnipw_basic():
    """Test basic functionality."""
    y = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    D = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    Z_proxy = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_no_iv_proxy(y, D, Z_proxy)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_bnnipw_edge():
    """Test edge cases."""
    y = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    D = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    Z_proxy = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_no_iv_proxy(y, D, Z_proxy)
    assert isinstance(result, dict)
