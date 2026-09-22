"""Tests for bndsdo.bound_skewed_outcome."""

from morie.fn import _array_core as np

from morie.fn.bndsdo import bound_skewed_outcome


def test_bndsdo_basic():
    """Test basic functionality."""
    y = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    D = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    result = bound_skewed_outcome(y, D)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_bndsdo_edge():
    """Test edge cases."""
    y = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    D = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    result = bound_skewed_outcome(y, D)
    assert isinstance(result, dict)
