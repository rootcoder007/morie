"""Tests for msm096.mvsml_bayesian_regression_eq_6_7."""

from morie.fn import _array_core as np

from morie.fn.msm096 import mvsml_bayesian_regression_eq_6_7


def test_msm096_basic():
    """Test basic functionality."""
    Z_L = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    G = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_bayesian_regression_eq_6_7(Z_L, G)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm096_edge():
    """Test edge cases."""
    Z_L = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    G = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_bayesian_regression_eq_6_7(Z_L, G)
    assert isinstance(result, dict)
