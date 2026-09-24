"""Tests for msm130.mvsml_categorical_count_eq_8_3."""

from morie.fn import _array_core as np

from morie.fn.msm130 import mvsml_categorical_count_eq_8_3


def test_msm130_basic():
    """Test basic functionality."""
    K = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    y = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = mvsml_categorical_count_eq_8_3(K, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm130_edge():
    """Test edge cases."""
    K = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    y = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = mvsml_categorical_count_eq_8_3(K, y)
    assert isinstance(result, dict)
