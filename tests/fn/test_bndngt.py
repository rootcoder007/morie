"""Tests for bndngt.bound_neg_treatment."""

from morie.fn import _array_core as np

from morie.fn.bndngt import bound_neg_treatment


def test_bndngt_basic():
    """Test basic functionality."""
    y = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    D = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    y_min = 0
    result = bound_neg_treatment(y, D, y_min)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_bndngt_edge():
    """Test edge cases."""
    y = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    D = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    y_min = 0
    result = bound_neg_treatment(y, D, y_min)
    assert isinstance(result, dict)
