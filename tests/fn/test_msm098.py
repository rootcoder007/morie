"""Tests for msm098.mvsml_bayesian_regression_pt2_eq_7_5."""

from morie.fn import _array_core as np

from morie.fn.msm098 import mvsml_bayesian_regression_pt2_eq_7_5


def test_msm098_basic():
    """Test basic functionality."""
    n = 5
    X_E = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Z_L = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_bayesian_regression_pt2_eq_7_5(n, X_E, Z_L)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm098_edge():
    """Test edge cases."""
    n = 5
    X_E = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Z_L = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_bayesian_regression_pt2_eq_7_5(n, X_E, Z_L)
    assert isinstance(result, dict)
