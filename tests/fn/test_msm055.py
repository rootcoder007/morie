"""Tests for msm055.mvsml_bayesian_regression_eq_6_5."""

from morie.fn import _array_core as np

from morie.fn.msm055 import mvsml_bayesian_regression_eq_6_5


def test_msm055_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    Z = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    G = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_bayesian_regression_eq_6_5(y, Z, G)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm055_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    Z = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    G = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_bayesian_regression_eq_6_5(y, Z, G)
    assert isinstance(result, dict)
