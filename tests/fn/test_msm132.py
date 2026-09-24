"""Tests for msm132.mvsml_categorical_count_eq_8_5."""

from morie.fn import _array_core as np

from morie.fn.msm132 import mvsml_categorical_count_eq_8_5


def test_msm132_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_categorical_count_eq_8_5(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm132_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_categorical_count_eq_8_5(X)
    assert isinstance(result, dict)
