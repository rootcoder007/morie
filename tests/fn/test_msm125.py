"""Tests for msm125.mvsml_categorical_count_eq_8_2."""

from morie.fn import _array_core as np

from morie.fn.msm125 import mvsml_categorical_count_eq_8_2


def test_msm125_basic():
    """Test basic functionality."""
    K_new = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    beta = 0.1
    result = mvsml_categorical_count_eq_8_2(K_new, beta)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm125_edge():
    """Test edge cases."""
    K_new = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    beta = 0.1
    result = mvsml_categorical_count_eq_8_2(K_new, beta)
    assert isinstance(result, dict)
