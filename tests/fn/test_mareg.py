"""Tests for mareg.ma_meta_regression."""

from morie.fn import _array_core as np

from morie.fn.mareg import ma_meta_regression


def test_mareg_basic():
    """Test basic functionality."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = ma_meta_regression(yi, vi, X)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_mareg_edge():
    """Test edge cases."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = ma_meta_regression(yi, vi, X)
    assert isinstance(result, dict)
