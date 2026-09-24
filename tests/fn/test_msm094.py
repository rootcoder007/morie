"""Tests for msm094.mvsml_bayesian_regression_pt2_eq_7_1."""

from morie.fn import _array_core as np

from morie.fn.msm094 import mvsml_bayesian_regression_pt2_eq_7_1


def test_msm094_basic():
    """Test basic functionality."""
    y = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_bayesian_regression_pt2_eq_7_1(y, X)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm094_edge():
    """Test edge cases."""
    y = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_bayesian_regression_pt2_eq_7_1(y, X)
    assert isinstance(result, dict)
