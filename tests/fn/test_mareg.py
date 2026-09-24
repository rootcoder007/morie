"""Tests for mareg.ma_meta_regression."""

from morie.fn import _array_core as np

from morie.fn.mareg import ma_meta_regression


def test_mareg_basic():
    """Test basic functionality."""
    yi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    vi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    X = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = ma_meta_regression(yi, vi, X)
    assert isinstance(result, dict)
    assert "beta" in result


def test_mareg_edge():
    """Test edge cases."""
    yi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    vi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    X = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = ma_meta_regression(yi, vi, X)
    assert isinstance(result, dict)
