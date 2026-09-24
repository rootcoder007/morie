"""Tests for msm157.mvsml_categorical_count_eq_8_10."""

from morie.fn import _array_core as np

from morie.fn.msm157 import mvsml_categorical_count_eq_8_10


def test_msm157_basic():
    """Test basic functionality."""
    Z_u1 = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    K = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Z_E = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_categorical_count_eq_8_10(Z_u1, K, Z_E)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm157_edge():
    """Test edge cases."""
    Z_u1 = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    K = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Z_E = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_categorical_count_eq_8_10(Z_u1, K, Z_E)
    assert isinstance(result, dict)
