"""Tests for bndnvg.bound_naive_gross."""

from morie.fn import _array_core as np

from morie.fn.bndnvg import bound_naive_gross


def test_bndnvg_basic():
    """Test basic functionality."""
    y = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    D = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    result = bound_naive_gross(y, D)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_bndnvg_edge():
    """Test edge cases."""
    y = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    D = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    result = bound_naive_gross(y, D)
    assert isinstance(result, dict)
