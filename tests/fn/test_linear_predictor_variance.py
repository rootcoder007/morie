"""Tests for linear_predictor_variance.linear_predictor_variance."""

import math

from morie.fn import _array_core as np

from morie.fn.linear_predictor_variance import (
    linear_predictor_variance,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e16_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    xs = rng.normal(0, 1, 4)
    cov = np.eye(4)
    result = linear_predictor_variance(xs, cov)
    assert isinstance(result, dict)
    assert "value" in result
    value = result["value"]
    assert isinstance(value, (int, float))
    assert math.isfinite(value)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e16_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    xs = np.ones(4)
    cov = np.zeros((4, 4))
    result = linear_predictor_variance(xs, cov)
    assert isinstance(result, dict)
    assert "value" in result
    value = result["value"]
    assert isinstance(value, (int, float))
    assert math.isfinite(value)
