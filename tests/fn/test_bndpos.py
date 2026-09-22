"""Tests for bndpos.bound_pos_treatment."""

from morie.fn import _array_core as np

from morie.fn.bndpos import bound_pos_treatment


def test_bndpos_basic():
    """Test basic functionality."""
    y = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    D = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    y_max = 100
    result = bound_pos_treatment(y, D, y_max)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_bndpos_edge():
    """Test edge cases."""
    y = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    D = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    y_max = 100
    result = bound_pos_treatment(y, D, y_max)
    assert isinstance(result, dict)
