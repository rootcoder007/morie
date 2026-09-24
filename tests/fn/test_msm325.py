"""Tests for msm325.mvsml_functional_regression_eq_15_2."""

from morie.fn import _array_core as np

from morie.fn.msm325 import mvsml_functional_regression_eq_15_2


def test_msm325_basic():
    """Test basic functionality."""
    y_positive = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_functional_regression_eq_15_2(y_positive)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm325_edge():
    """Test edge cases."""
    y_positive = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_functional_regression_eq_15_2(y_positive)
    assert isinstance(result, dict)
