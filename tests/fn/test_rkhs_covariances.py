"""Tests for rkhs_covariances.rkhs_covariances."""

from morie.fn import _array_core as np

from morie.fn.rkhs_covariances import rkhs_covariances


def test_msm063_basic():
    """Test basic functionality."""
    where = np.random.default_rng(42).normal(0, 1, 100)
    XE = np.random.default_rng(42).normal(0, 1, 100)
    XEM = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    design = np.random.default_rng(42).normal(0, 1, 100)
    result = rkhs_covariances(where, XE, XEM, are, the, design)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm063_edge():
    """Test edge cases."""
    where = np.random.default_rng(42).normal(0, 1, 100)
    XE = np.random.default_rng(42).normal(0, 1, 100)
    XEM = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    design = np.random.default_rng(42).normal(0, 1, 100)
    result = rkhs_covariances(where, XE, XEM, are, the, design)
    assert isinstance(result, dict)
