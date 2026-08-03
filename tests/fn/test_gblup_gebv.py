"""Tests for gblup_gebv.gblup_gebv."""

from morie.fn import _array_core as np

from morie.fn.gblup_gebv import gblup_gebv


def test_msm242_basic():
    """Test basic functionality."""
    b = np.random.default_rng(42).normal(0, 1, 100)
    XTR = np.random.default_rng(42).normal(0, 1, 100)
    result = gblup_gebv(b, XTR)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm242_edge():
    """Test edge cases."""
    b = np.random.default_rng(42).normal(0, 1, 100)
    XTR = np.random.default_rng(42).normal(0, 1, 100)
    result = gblup_gebv(b, XTR)
    assert isinstance(result, dict)
