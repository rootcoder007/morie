"""Tests for msm154.mvsml_categorical_count_eq_8_11."""

from morie.fn import _array_core as np

from morie.fn.msm154 import mvsml_categorical_count_eq_8_11


def test_msm154_basic():
    """Test basic functionality."""
    K = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = mvsml_categorical_count_eq_8_11(K)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm154_edge():
    """Test edge cases."""
    K = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = mvsml_categorical_count_eq_8_11(K)
    assert isinstance(result, dict)
