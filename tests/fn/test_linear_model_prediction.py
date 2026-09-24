"""Tests for linear_model_prediction.linear_model_prediction."""

from morie.fn import _array_core as np

from morie.fn.linear_model_prediction import (
    linear_model_prediction,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r16e1_basic():
    """Test basic functionality."""
    beta0 = 0.5
    beta1 = 0.5
    x = 0.5
    result = linear_model_prediction(beta0, beta1, x)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r16e1_edge():
    """Test edge cases."""
    beta0 = 0.5
    beta1 = 0.5
    x = 0.5
    result = linear_model_prediction(beta0, beta1, x)
    assert isinstance(result, dict)
