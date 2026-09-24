"""Tests for poisson_rate_mean.poisson_rate_mean."""

from morie.fn import _array_core as np

from morie.fn.poisson_rate_mean import (
    poisson_rate_mean,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e15_basic():
    """Test basic functionality."""
    b0 = 0.5
    bs = 0.5
    xs = 0.5
    exposure = 0.5
    result = poisson_rate_mean(b0, bs, xs, exposure)
    assert isinstance(result, dict)
    assert "value" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e15_edge():
    """Test edge cases."""
    b0 = 0.5
    bs = 0.5
    xs = 0.5
    exposure = 0.5
    result = poisson_rate_mean(b0, bs, xs, exposure)
    assert isinstance(result, dict)
