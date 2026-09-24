"""Tests for rkhs_covariances.rkhs_covariances."""

from morie.fn import _array_core as np

from morie.fn.rkhs_covariances import rkhs_covariances


def test_msm063_basic():
    """Test basic functionality."""
    Z_L = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    G = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = rkhs_covariances(Z_L, G)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm063_edge():
    """Test edge cases."""
    Z_L = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    G = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = rkhs_covariances(Z_L, G)
    assert isinstance(result, dict)
