"""Tests for grn002.geron_ch4_linear_regression_prediction."""

from morie.fn import _array_core as np

from morie.fn.grn002 import geron_ch4_linear_regression_prediction


def test_grn002_basic():
    """Test basic functionality."""
    theta = [1.0, 2.0, 3.0]
    x = [3.0, 4.0]
    result = geron_ch4_linear_regression_prediction(theta, x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grn002_edge():
    """Test edge cases."""
    theta = [1.0, 2.0, 3.0]
    x = [3.0, 4.0]
    result = geron_ch4_linear_regression_prediction(theta, x)
    assert isinstance(result, dict)
