"""Tests for msm117.mvsml_bayesian_regression_pt2_eq_7_7."""

from morie.fn import _array_core as np

from morie.fn.msm117 import mvsml_bayesian_regression_pt2_eq_7_7


def test_msm117_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    beta0 = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    beta = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_bayesian_regression_pt2_eq_7_7(X, y, beta0, beta)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm117_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    beta0 = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    beta = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_bayesian_regression_pt2_eq_7_7(X, y, beta0, beta)
    assert isinstance(result, dict)
