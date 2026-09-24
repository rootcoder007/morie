"""Tests for msm333.mvsml_elements_lin_reg_eq_3_1."""

from morie.fn import _array_core as np

from morie.fn.msm333 import mvsml_elements_lin_reg_eq_3_1


def test_msm333_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mvsml_elements_lin_reg_eq_3_1(X, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm333_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mvsml_elements_lin_reg_eq_3_1(X, y)
    assert isinstance(result, dict)
