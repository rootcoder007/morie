"""Tests for msm100.mvsml_bayesian_regression_pt2_eq_7_4."""

from morie.fn import _array_core as np

from morie.fn.msm100 import mvsml_bayesian_regression_pt2_eq_7_4


def test_msm100_basic():
    """Test basic functionality."""
    y = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    G = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    result = mvsml_bayesian_regression_pt2_eq_7_4(y, G)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm100_edge():
    """Test edge cases."""
    y = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    G = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    result = mvsml_bayesian_regression_pt2_eq_7_4(y, G)
    assert isinstance(result, dict)
