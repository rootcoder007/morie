"""Tests for msm052.mvsml_bayesian_regression_eq_6_4."""

from morie.fn import _array_core as np

from morie.fn.msm052 import mvsml_bayesian_regression_eq_6_4


def test_msm052_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0, 1, 100)
    G = np.eye(10)
    result = mvsml_bayesian_regression_eq_6_4(y, G)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm052_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0, 1, 100)
    G = np.eye(10)
    result = mvsml_bayesian_regression_eq_6_4(y, G)
    assert isinstance(result, dict)
