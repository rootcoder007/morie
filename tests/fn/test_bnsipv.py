"""Tests for bnsipv.bound_iv_partial."""

from morie.fn import _array_core as np

from morie.fn.bnsipv import bound_iv_partial


def test_bnsipv_basic():
    """Test basic functionality."""
    y = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    D = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    Z = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    result = bound_iv_partial(y, D, Z)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_bnsipv_edge():
    """Test edge cases."""
    y = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    D = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    Z = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    result = bound_iv_partial(y, D, Z)
    assert isinstance(result, dict)
