"""Tests for snp_blup_gebv.snp_blup_gebv."""

from morie.fn import _array_core as np

from morie.fn.snp_blup_gebv import snp_blup_gebv


def test_msm243_basic():
    """Test basic functionality."""
    b = np.random.default_rng(42).normal(0, 1, 100)
    XTR = np.random.default_rng(42).normal(0, 1, 100)
    result = snp_blup_gebv(b, XTR)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm243_edge():
    """Test edge cases."""
    b = np.random.default_rng(42).normal(0, 1, 100)
    XTR = np.random.default_rng(42).normal(0, 1, 100)
    result = snp_blup_gebv(b, XTR)
    assert isinstance(result, dict)
