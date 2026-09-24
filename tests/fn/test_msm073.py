"""Tests for msm073.mvsml_bayesian_regression_eq_6_8."""

from morie.fn import _array_core as np

from morie.fn.msm073 import mvsml_bayesian_regression_eq_6_8


def test_msm073_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Z1 = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    G = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_bayesian_regression_eq_6_8(Y, Z1, G)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm073_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Z1 = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    G = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_bayesian_regression_eq_6_8(Y, Z1, G)
    assert isinstance(result, dict)
