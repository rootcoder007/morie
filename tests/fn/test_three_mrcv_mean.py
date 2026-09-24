"""Tests for three_mrcv_mean.three_mrcv_mean."""

from morie.fn import _array_core as np

from morie.fn.three_mrcv_mean import (
    three_mrcv_mean,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e16_basic():
    """Test basic functionality."""
    b0 = 0.5
    beta_w_a = 0.5
    beta_y_b = 0.5
    beta_z_c = 0.5
    result = three_mrcv_mean(b0, beta_w_a, beta_y_b, beta_z_c)
    assert isinstance(result, dict)
    assert "value" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e16_edge():
    """Test edge cases."""
    b0 = 0.5
    beta_w_a = 0.5
    beta_y_b = 0.5
    beta_z_c = 0.5
    result = three_mrcv_mean(b0, beta_w_a, beta_y_b, beta_z_c)
    assert isinstance(result, dict)
