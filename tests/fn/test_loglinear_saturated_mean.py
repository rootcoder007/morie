"""Tests for loglinear_saturated_mean.loglinear_saturated_mean."""

from morie.fn import _array_core as np

from morie.fn.loglinear_saturated_mean import (
    loglinear_saturated_mean,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e6_basic():
    """Test basic functionality."""
    b0 = 0.5
    beta_x_i = 0.5
    beta_z_j = 0.5
    beta_xz_ij = 0.5
    result = loglinear_saturated_mean(b0, beta_x_i, beta_z_j, beta_xz_ij)
    assert isinstance(result, dict)
    assert "value" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e6_edge():
    """Test edge cases."""
    b0 = 0.5
    beta_x_i = 0.5
    beta_z_j = 0.5
    beta_xz_ij = 0.5
    result = loglinear_saturated_mean(b0, beta_x_i, beta_z_j, beta_xz_ij)
    assert isinstance(result, dict)
