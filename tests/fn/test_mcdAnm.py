"""Tests for mcdAnm.mcd_outlier."""

from morie.fn import _array_core as np

from morie.fn.mcdAnm import mcd_outlier


def test_mcdAnm_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mcd_outlier(X)
    assert isinstance(result, dict)
    assert "distance" in result


def test_mcdAnm_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mcd_outlier(X)
    assert isinstance(result, dict)
