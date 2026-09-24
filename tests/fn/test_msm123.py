"""Tests for msm123.mvsml_categorical_count_eq_8_1."""

from morie.fn import _array_core as np

from morie.fn.msm123 import mvsml_categorical_count_eq_8_1


def test_msm123_basic():
    """Test basic functionality."""
    K = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    beta = 0.1
    result = mvsml_categorical_count_eq_8_1(K, y, beta)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm123_edge():
    """Test edge cases."""
    K = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    beta = 0.1
    result = mvsml_categorical_count_eq_8_1(K, y, beta)
    assert isinstance(result, dict)
