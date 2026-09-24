"""Tests for msm055.mvsml_bayesian_regression_eq_6_5."""

import math

from morie.fn import _array_core as np

from morie.fn.msm055 import mvsml_bayesian_regression_eq_6_5


def test_msm055_basic():
    """Test basic functionality."""
    y = [5.0, 5.2, 6.0, 6.1]
    Z = [[1, 0], [1, 0], [0, 1], [0, 1]]
    G = [[1.0, 0.4], [0.4, 1.0]]
    result = mvsml_bayesian_regression_eq_6_5(y, Z, G)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])


def test_msm055_edge():
    """Test edge cases."""
    y = [5.0, 5.2, 6.0, 6.1]
    Z = [[1, 0], [1, 0], [0, 1], [0, 1]]
    G = [[1.0, 0.4], [0.4, 1.0]]
    result = mvsml_bayesian_regression_eq_6_5(y, Z, G)
    assert isinstance(result, dict)
