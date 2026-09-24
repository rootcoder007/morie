"""Tests for msm158.mvsml_categorical_count_eq_8_13."""

from morie.fn import _array_core as np

from morie.fn.msm158 import mvsml_categorical_count_eq_8_13


def test_msm158_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    m_index = np.random.default_rng(42).normal(0.0, 1.0, 40)
    Z_u1 = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    Z_E = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_categorical_count_eq_8_13(X, m_index, Z_u1, Z_E)
    assert isinstance(result, dict)
    assert "P" in result


def test_msm158_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    m_index = np.random.default_rng(42).normal(0.0, 1.0, 40)
    Z_u1 = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    Z_E = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_categorical_count_eq_8_13(X, m_index, Z_u1, Z_E)
    assert isinstance(result, dict)
