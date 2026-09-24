"""Tests for msm121.mvsml_bayesian_regression_pt2_eq_7_6."""

from morie.fn import _array_core as np

from morie.fn.msm121 import mvsml_bayesian_regression_pt2_eq_7_6


def test_msm121_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    beta0 = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    beta = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_bayesian_regression_pt2_eq_7_6(X, beta0, beta)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm121_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    beta0 = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    beta = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_bayesian_regression_pt2_eq_7_6(X, beta0, beta)
    assert isinstance(result, dict)
