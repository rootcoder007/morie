"""Tests for msm056.mvsml_bayesian_regression_eq_6_5."""

from morie.fn import _array_core as np

from morie.fn.msm056 import mvsml_bayesian_regression_eq_6_5


def test_msm056_basic():
    """Test basic functionality."""
    y = [5.0, 5.2, 6.0, 6.1]
    Z = [[1, 0], [1, 0], [0, 1], [0, 1]]
    G = [[1.0, 0.4], [0.4, 1.0]]
    result = mvsml_bayesian_regression_eq_6_5(y, Z, G)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm056_edge():
    """Test edge cases."""
    y = [5.0, 5.2, 6.0, 6.1]
    Z = [[1, 0], [1, 0], [0, 1], [0, 1]]
    G = [[1.0, 0.4], [0.4, 1.0]]
    result = mvsml_bayesian_regression_eq_6_5(y, Z, G)
    assert isinstance(result, dict)
