"""Tests for miprgr.mip_branch_bound."""

from morie.fn import _array_core as np

from morie.fn.miprgr import mip_branch_bound


def test_miprgr_basic():
    """Test basic functionality."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    b = np.random.default_rng(42).normal(0.0, 1.0, 40)
    c = np.random.default_rng(42).normal(0.0, 1.0, 40)
    integer_vars = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = mip_branch_bound(A, b, c, integer_vars)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_miprgr_edge():
    """Test edge cases."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    b = np.random.default_rng(42).normal(0.0, 1.0, 40)
    c = np.random.default_rng(42).normal(0.0, 1.0, 40)
    integer_vars = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = mip_branch_bound(A, b, c, integer_vars)
    assert isinstance(result, dict)
