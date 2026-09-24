"""Tests for msm046.mvsml_bayesian_regression_eq_6_3."""

from morie.fn import _array_core as np

from morie.fn.msm046 import mvsml_bayesian_regression_eq_6_3


def test_msm046_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_bayesian_regression_eq_6_3(y, X)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm046_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_bayesian_regression_eq_6_3(y, X)
    assert isinstance(result, dict)
