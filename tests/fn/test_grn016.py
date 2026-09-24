"""Tests for grn016.geron_ch4_logistic_regression_prediction."""

from morie.fn import _array_core as np

from morie.fn.grn016 import geron_ch4_logistic_regression_prediction


def test_grn016_basic():
    """Test basic functionality."""
    p_hat = [0.0, 0.4999, 0.5, 1.0]
    result = geron_ch4_logistic_regression_prediction(p_hat)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grn016_edge():
    """Test edge cases."""
    p_hat = [0.0, 0.4999, 0.5, 1.0]
    result = geron_ch4_logistic_regression_prediction(p_hat)
    assert isinstance(result, dict)
