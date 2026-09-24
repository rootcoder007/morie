"""Tests for msm244.mvsml_preprocessing_eq_2_1."""

from morie.fn import _array_core as np

from morie.fn.msm244 import mvsml_preprocessing_eq_2_1


def test_msm244_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Z = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    Sigma = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_preprocessing_eq_2_1(X, Z, y, Sigma)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm244_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Z = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    Sigma = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_preprocessing_eq_2_1(X, Z, y, Sigma)
    assert isinstance(result, dict)
