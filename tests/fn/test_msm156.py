"""Tests for msm156.mvsml_categorical_count_eq_8_12."""

from morie.fn import _array_core as np

from morie.fn.msm156 import mvsml_categorical_count_eq_8_12


def test_msm156_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    m_index = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = mvsml_categorical_count_eq_8_12(X, m_index)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm156_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    m_index = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = mvsml_categorical_count_eq_8_12(X, m_index)
    assert isinstance(result, dict)
