"""Tests for msm142.mvsml_categorical_count_eq_8_9."""

from morie.fn import _array_core as np

from morie.fn.msm142 import mvsml_categorical_count_eq_8_9


def test_msm142_basic():
    """Test basic functionality."""
    Z = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    K = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_categorical_count_eq_8_9(Z, K)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm142_edge():
    """Test edge cases."""
    Z = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    K = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_categorical_count_eq_8_9(Z, K)
    assert isinstance(result, dict)
