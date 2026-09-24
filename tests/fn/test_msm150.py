"""Tests for msm150.mvsml_categorical_count_eq_8_8."""

from morie.fn import _array_core as np

from morie.fn.msm150 import mvsml_categorical_count_eq_8_8


def test_msm150_basic():
    """Test basic functionality."""
    y = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    K = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    result = mvsml_categorical_count_eq_8_8(y, K)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm150_edge():
    """Test edge cases."""
    y = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    K = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    result = mvsml_categorical_count_eq_8_8(y, K)
    assert isinstance(result, dict)
