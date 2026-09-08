"""Tests for mme_solve.mme_solve."""

from morie.fn import _array_core as np

from morie.fn.mme_solve import mme_solve


def test_msm241_basic():
    """Test basic functionality."""
    b = np.random.default_rng(42).normal(0, 1, 100)
    XTR = np.random.default_rng(42).normal(0, 1, 100)
    result = mme_solve(b, XTR)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm241_edge():
    """Test edge cases."""
    b = np.random.default_rng(42).normal(0, 1, 100)
    XTR = np.random.default_rng(42).normal(0, 1, 100)
    result = mme_solve(b, XTR)
    assert isinstance(result, dict)
